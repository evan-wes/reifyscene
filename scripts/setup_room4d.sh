#!/usr/bin/env bash
set -euo pipefail

# Reproducible installer for the current room4d prototype.
#
# What this installs today:
# - the local room4d package in editable mode
# - Python dependencies declared in pyproject.toml
# - optional detectron2 from its official v0.6 release tag
# - optional VGGT from the official repo at a pinned commit, cloned into
#   third_party/vggt and installed into the active environment
# - optional CoTracker from the official repo at a pinned commit, cloned into
#   third_party/co-tracker and installed into the active environment
# - optional VGGT-Long from the official repo at a pinned commit, cloned into
#   third_party/VGGT-Long and installed into the active environment
# - optional calibration dependencies: pycolmap pinned to a COLMAP release,
#   plus hloc cloned into third_party/Hierarchical-Localization at a pinned tag
#
# Environment behavior:
# - By default, this script expects to use a conda environment.
# - You may provide --conda-env NAME to require / activate a specific env.
# - If no conda env is active and none can be activated, the script exits.
# - There is NO silent fallback to venv.
# - To explicitly use a repo-local venv instead, pass --use-venv.

# ------------------------------------------------------------------------------
# Pinned external dependency refs
# ------------------------------------------------------------------------------

DETECTRON2_REPO="https://github.com/facebookresearch/detectron2.git"
DETECTRON2_TAG="v0.6"

VGGT_REPO="https://github.com/facebookresearch/vggt.git"
VGGT_COMMIT="44b3afbd1869d8bde4894dd8ea1e293112dd5eba"

COTRACKER_REPO="https://github.com/facebookresearch/co-tracker.git"
COTRACKER_COMMIT="82e02e8029753ad4ef13cf06be7f4fc5facdda4d"

# SAM 2 (Apache-2.0, ungated) — the FP2 `segment` top rung: class-agnostic
# automatic mask generation + video mask propagation (persistent instance IDs).
# Pinned to main HEAD at vendoring time. The sm_120 CUDA connected-components
# extension is OPTIONAL (Python fallback exists), so the install skips it
# (SAM2_BUILD_CUDA=0) to stay toolchain-light. The SAM 2.1 Hiera-Large checkpoint
# ships at a direct fbaipublicfiles URL (like the tapnet checkpoints), so the
# fetch is fully scripted + idempotent. ~900 MB.
SAM2_REPO="https://github.com/facebookresearch/sam2.git"
SAM2_COMMIT="2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM2_CKPT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
SAM2_CKPT_MIN_MB="${SAM2_CKPT_MIN_MB:-700}"

# DINOv2-L (Apache-2.0, ungated) — the FP3 `reid` top rung: frozen ViT appearance
# features for cross-occlusion identity stitching, above the ResNet18 floor. Loaded
# through HuggingFace transformers (`AutoModel`), so install = the transformers pip
# dep + pre-caching the weights from the HF hub (~1.2 GB, no manual URL — unlike the
# sam2 checkpoint). The gated DINOv3-L (facebook/dinov3-vitl16-pretrain-lvd1689m) is
# a drop-in model_id swap once HF access is granted: same AutoModel call, but needs
# `hf auth login` + a transformers build with Dinov3 support.
DINOV2_MODEL_ID="${DINOV2_MODEL_ID:-facebook/dinov2-large}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers>=4.44}"

# SigLIP2 (Google, Apache-2.0, ungated) — the FP4 `semantics` top rung: a frozen
# ViT image+text encoder for per-tracklet open-vocab category (zero-shot match vs a
# candidate vocabulary). Loaded through transformers (`AutoModel`/`AutoProcessor`),
# the same idiom as DINOv2; install = the (shared) transformers pip dep + pre-caching
# the weights (~1.5 GB, no manual URL). Fixed-resolution variant -> standard pixel_values.
SIGLIP2_MODEL_ID="${SIGLIP2_MODEL_ID:-google/siglip2-so400m-patch14-384}"

# Sapiens (Meta, CC-BY-NC-4.0 non-commercial, ungated) — the FP5 `humans` dense top
# rung: pose / body-part seg / depth / normals via TorchScript checkpoints, loaded
# directly with torch.jit (no transformers). Install = pre-caching the 4 ungated HF
# checkpoints for the chosen size into the HF hub cache (~5 GB at 0.6B). The repo +
# filename list lives in room4d.perception.humans.sapiens_wrapper._CHECKPOINTS.
SAPIENS_SIZE="${SAPIENS_SIZE:-0.6b}"

VGGT_LONG_REPO="https://github.com/DengKaiCQ/VGGT-Long.git"
VGGT_LONG_COMMIT="c160869d1d99c96bb227f414afb3bc68c29c9a76"

HLOC_REPO="https://github.com/cvg/Hierarchical-Localization.git"
HLOC_TAG="v1.4"
COLMAP_UPSTREAM_TAG="3.13.0"
PYCOLMAP_VERSION="3.13.0"

# gsplat is built from source for sm_120 (no prebuilt wheel exists for this
# torch/CUDA line). Pinned to the same Blackwell-proven commit as
# setup_reference_tool.sh; the reference env built it against CUDA 12.8 —
# this env builds the same ref against CUDA 13.0.
GSPLAT_REPO="https://github.com/nerfstudio-project/gsplat.git"
GSPLAT_REF="4e52698e45eaaed929ed3a5065e96a688d085df6"
GSPLAT_ARCH_LIST="12.0"   # TORCH_CUDA_ARCH_LIST for the 5090 (sm_120)

# ffmpeg CLI for dataset prep: assembling image-sequence datasets (DyCheck,
# DAVIS) into the mp4s the pipeline ingests. room4d's own decode path uses
# OpenCV and does NOT need this — it is a workflow convenience. conda-forge
# ships a GPL-variant build (build string `gpl_*`) that includes libx264,
# required for `-crf 0` lossless H.264.
FFMPEG_VERSION="${FFMPEG_VERSION:-8.1.1}"

# faiss (conda-forge CPU build) for VGGT-Long loop closure's place-recognition
# retrieval. Stripped from VGGT-Long's pip requirements (faiss-gpu needs a CUDA
# build); the CPU build is sufficient for retrieval over a clip's frames.
FAISS_VERSION="${FAISS_VERSION:-1.9.0}"

REQUIRED_GXX_MAJOR=9
CONDA_GCC_VERSION="${CONDA_GCC_VERSION:-11}"
# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"
USE_VENV=0
UNLOCKED=0

# Full pip-layer lockfile (see scripts/gen_env_locks.sh). Installed before the
# editable room4d install so the transitive dependency tree replays the
# known-good snapshot instead of re-resolving. torch +cu130 wheels only exist
# on the pytorch index, so the lock install needs it on a fresh env.
ROOM4D_LOCK_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/locks/room4d.lock"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"

INSTALL_DETECTRON2=0
INSTALL_VGGT=0
INSTALL_COTRACKER=0
INSTALL_SAM2=0
INSTALL_DINOV2=0
INSTALL_SIGLIP2=0
INSTALL_SAPIENS=0
INSTALL_VGGT_LONG=0
INSTALL_GSPLAT=1
INSTALL_DEV=0
INSTALL_VISION=1
INSTALL_GEOMETRY=1
INSTALL_CALIBRATION=1
INSTALL_POSE_REFINE=1
INSTALL_FFMPEG=1
FORCE_VGGT_LONG_WEIGHTS=0

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"
VGGT_DIR="${THIRD_PARTY_DIR}/vggt"
COTRACKER_DIR="${THIRD_PARTY_DIR}/co-tracker"
SAM2_DIR="${THIRD_PARTY_DIR}/sam2"
SAM2_CKPT_DIR="${SAM2_DIR}/checkpoints"
SAM2_CKPT="${SAM2_CKPT_DIR}/sam2.1_hiera_large.pt"
VGGT_LONG_DIR="${THIRD_PARTY_DIR}/VGGT-Long"
HLOC_DIR="${THIRD_PARTY_DIR}/Hierarchical-Localization"
VGGT_LONG_WEIGHTS_DIR="${VGGT_LONG_DIR}/weights"
VGGT_LONG_WEIGHTS_STAMP="${VGGT_LONG_WEIGHTS_DIR}/.room4d_download_complete"
VGGT_LONG_WEIGHTS_MIN_MB="${VGGT_LONG_WEIGHTS_MIN_MB:-8000}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"; shift 2 ;;
    --venv)
      VENV_DIR="$2"; shift 2 ;;
    --conda-env)
      CONDA_ENV_NAME="$2"; shift 2 ;;
    --use-venv)
      USE_VENV=1; shift ;;
    --with-detectron2)
      INSTALL_DETECTRON2=1; shift ;;
    --with-vggt)
      INSTALL_VGGT=1; shift ;;
    --with-cotracker)
      INSTALL_COTRACKER=1; shift ;;
    --with-sam2)
      INSTALL_SAM2=1; shift ;;
    --with-dinov2)
      INSTALL_DINOV2=1; shift ;;
    --with-siglip2)
      INSTALL_SIGLIP2=1; shift ;;
    --with-sapiens)
      INSTALL_SAPIENS=1; shift ;;
    --with-vggt-long)
      INSTALL_VGGT_LONG=1; shift ;;
    --with-gsplat)
      INSTALL_GSPLAT=1; shift ;;
    --no-gsplat)
      INSTALL_GSPLAT=0; shift ;;
    --force-vggt-long-weights)
      FORCE_VGGT_LONG_WEIGHTS=1; shift ;;
    --with-dev)
      INSTALL_DEV=1; shift ;;
    --no-vision)
      INSTALL_VISION=0; shift ;;
    --no-geometry)
      INSTALL_GEOMETRY=0; shift ;;
    --no-calibration)
      INSTALL_CALIBRATION=0; shift ;;
    --no-pose-refine)
      INSTALL_POSE_REFINE=0; shift ;;
    --no-ffmpeg)
      INSTALL_FFMPEG=0; shift ;;
    --unlocked)
      UNLOCKED=1; shift ;;
    -h|--help)
      cat <<'EOF'
Usage: scripts/setup_room4d.sh [options]

Environment options:
  --conda-env NAME        Require / activate this conda environment
  --use-venv              Explicitly use a repo-local venv instead of conda
  --python PATH           Python executable to use (default: python3)
  --venv PATH             Virtualenv path (default: .venv)

Install options:
  --with-detectron2       Install detectron2 from official tag v0.6
  --with-vggt             Clone/install VGGT into third_party/vggt at pinned commit
  --with-cotracker        Clone/install CoTracker into third_party/co-tracker at pinned commit
  --with-sam2             Clone/install SAM 2 into third_party/sam2 at pinned commit
                          + download the sam2.1_hiera_large checkpoint (FP2 segment rung)
  --with-dinov2           Install HuggingFace transformers + pre-cache the DINOv2-L
                          weights (FP3 reid top rung; DINOv3-L swaps in by model_id)
  --with-siglip2          Install HuggingFace transformers + pre-cache the SigLIP2
                          weights (FP4 semantics top rung; ungated, open-vocab category)
  --with-sapiens          Pre-cache the Sapiens dense checkpoints (FP5 humans top rung:
                          pose/seg/depth/normals; ungated, CC-BY-NC-4.0 non-commercial)
  --with-vggt-long        Clone/install VGGT-Long into third_party/VGGT-Long at pinned commit
  --with-gsplat           Build gsplat from source for sm_120 at pinned commit
                          (default: on — the room4d renderer requires it)
  --no-gsplat             Skip the gsplat source build
  --force-vggt-long-weights
                          Re-download VGGT-Long weights even if already present
  --with-dev              Install dev tools (pytest, ruff, mypy)
  --no-vision             Skip vision extras
  --no-geometry           Skip geometry extras
  --no-calibration        Skip calibration extras and hloc install
  --no-pose-refine        Skip pose-refinement extras (pycolmap, hydra-core, omegaconf, scipy)
  --no-ffmpeg             Skip the ffmpeg (conda-forge) dataset-prep install
  --unlocked              Resolve dependencies fresh instead of installing the
                          pinned tree from scripts/locks/room4d.lock
                          (regenerate locks with scripts/gen_env_locks.sh)

Behavior:
  - Default mode is conda-only.
  - If no conda environment is active and none can be activated, the script exits.
  - There is no silent fallback to venv.
  - Use --use-venv if you explicitly want a repo-local virtualenv.
EOF
      exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2 ;;
  esac
done

if [[ $USE_VENV -eq 1 && -n "${CONDA_ENV_NAME}" ]]; then
  echo "Cannot use --use-venv and --conda-env together." >&2
  exit 2
fi

clone_or_update_checkout() {
  local repo_url="$1"
  local dest_dir="$2"
  local ref="$3"

  mkdir -p "$(dirname "$dest_dir")"

  if [[ ! -d "${dest_dir}/.git" ]]; then
    echo "Cloning ${repo_url} into ${dest_dir}..."
    git clone "${repo_url}" "${dest_dir}"
  fi

  echo "Fetching updates in ${dest_dir}..."
  git -C "${dest_dir}" fetch origin

  # Restore tracked files before checkout: this script deliberately patches
  # vendored trees after checkout (hloc pycolmap compat, VGGT-Long download
  # script), and a dirty tree would make the next ref change abort with
  # "local changes would be overwritten". Patches are reapplied fresh below,
  # so restoring here keeps every run deterministic (same pattern as
  # setup_reference_tool.sh).
  echo "Restoring tracked files in ${dest_dir} (patches are reapplied after checkout)..."
  git -C "${dest_dir}" checkout -- . 2>/dev/null || true

  echo "Checking out ${ref} in ${dest_dir}..."
  git -C "${dest_dir}" checkout "${ref}"
}

compiler_major_version() {
  local compiler="$1"
  if ! command -v "${compiler}" >/dev/null 2>&1; then
    echo ""
    return
  fi
  "${compiler}" -dumpfullversion -dumpversion 2>/dev/null | head -n 1 | cut -d. -f1
}

run_conda_install_relaxed() {
  # Conda activation/deactivation hooks are not always nounset-safe.
  # This script uses `set -u`, so temporarily relax that around conda install.
  local had_nounset=0
  case "$-" in
    *u*) had_nounset=1 ;;
  esac

  set +u
  conda install "$@"
  local rc=$?

  if [[ ${had_nounset} -eq 1 ]]; then
    set -u
  fi

  return ${rc}
}

python_one_liner() {
  local snippet="$1"
  python - <<PY
${snippet}
PY
}

detect_torch_cuda_version() {
  python_one_liner "import torch; print(torch.version.cuda or '')" 2>/dev/null || true
}

detect_torch_cuda_home() {
  python_one_liner "from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME or '')" 2>/dev/null || true
}

nvcc_cuda_version() {
  if ! command -v nvcc >/dev/null 2>&1; then
    return 0
  fi
  nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n 1
}

ensure_cuda_home_exports() {
  local cuda_home="$1"
  if [[ -z "${cuda_home}" ]]; then
    return 1
  fi

  export CUDA_HOME="${cuda_home}"

  if [[ -d "${CUDA_HOME}/bin" ]]; then
    case ":${PATH}:" in
      *":${CUDA_HOME}/bin:"*) ;;
      *) export PATH="${CUDA_HOME}/bin:${PATH}" ;;
    esac
  fi

  local lib_dir
  for lib_dir in \
    "${CUDA_HOME}/lib64" \
    "${CUDA_HOME}/lib" \
    "${CUDA_HOME}/targets/x86_64-linux/lib"
  do
    if [[ -d "${lib_dir}" ]]; then
      case ":${LD_LIBRARY_PATH:-}:" in
        *":${lib_dir}:"*) ;;
        *) export LD_LIBRARY_PATH="${lib_dir}:${LD_LIBRARY_PATH:-}" ;;
      esac
    fi
  done

  return 0
}

ensure_cuda_toolkit_for_torch_extensions() {
  local existing_cuda_home=""
  existing_cuda_home="$(detect_torch_cuda_home | tr -d '[:space:]')"

  if [[ -n "${existing_cuda_home}" && -x "${existing_cuda_home}/bin/nvcc" ]]; then
    ensure_cuda_home_exports "${existing_cuda_home}"
    echo "Using existing CUDA toolkit detected by torch: ${CUDA_HOME}"
    echo "  nvcc: $(command -v nvcc)"
    echo "  nvcc version: $(nvcc --version | tail -n 1)"
    return
  fi

  if [[ $USE_VENV -eq 1 ]]; then
    echo "This install step needs a CUDA toolkit with nvcc, but none was detected." >&2
    echo "In --use-venv mode the setup script does not install CUDA automatically." >&2
    echo "Either install a local CUDA toolkit manually and set CUDA_HOME, or use a conda env." >&2
    exit 1
  fi

  if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "No active conda environment detected while ensuring CUDA toolkit." >&2
    exit 1
  fi

  local torch_cuda=""
  torch_cuda="$(detect_torch_cuda_version | tr -d '[:space:]')"
  if [[ -z "${torch_cuda}" ]]; then
    echo "Could not determine torch.version.cuda from the active environment." >&2
    echo "Make sure torch is installed before requesting CUDA-extension builds." >&2
    exit 1
  fi

  local -a channels=()
  local patch
  for patch in 2 1 0; do
    channels+=("nvidia/label/cuda-${torch_cuda}.${patch}")
  done
  channels+=("nvidia")

  local installed=0
  local channel
  for channel in "${channels[@]}"; do
    echo "Attempting to install CUDA toolkit from conda channel: ${channel}"
    if run_conda_install_relaxed -y -c "${channel}" cuda-toolkit; then
      installed=1
      break
    fi
    echo "Conda install attempt from ${channel} did not succeed; trying next option..."
  done

  if [[ ${installed} -ne 1 ]]; then
    echo "Failed to install a CUDA toolkit into the active conda environment." >&2
    echo "Tried torch CUDA line ${torch_cuda} and a generic NVIDIA channel fallback." >&2
    exit 1
  fi

  ensure_cuda_home_exports "${CONDA_PREFIX}"

  local refreshed_cuda_home=""
  refreshed_cuda_home="$(detect_torch_cuda_home | tr -d '[:space:]')"
  if [[ -n "${refreshed_cuda_home}" ]]; then
    ensure_cuda_home_exports "${refreshed_cuda_home}"
  fi

  if ! command -v nvcc >/dev/null 2>&1; then
    echo "CUDA toolkit install completed, but nvcc is still not on PATH." >&2
    echo "CUDA_HOME=${CUDA_HOME:-unset}" >&2
    exit 1
  fi

  echo "Installed CUDA toolkit for extension builds:"
  echo "  torch.version.cuda=${torch_cuda}"
  echo "  CUDA_HOME=${CUDA_HOME}"
  echo "  nvcc=$(command -v nvcc)"
  echo "  nvcc version: $(nvcc --version | tail -n 1)"
}

vggt_long_weights_present() {
  if [[ ! -d "${VGGT_LONG_WEIGHTS_DIR}" ]]; then
    return 1
  fi

  if [[ -f "${VGGT_LONG_WEIGHTS_STAMP}" ]]; then
    return 0
  fi

  local total_mb=""
  total_mb="$(du -sm "${VGGT_LONG_WEIGHTS_DIR}" 2>/dev/null | awk '{print $1}')"
  if [[ -z "${total_mb}" || "${total_mb}" -lt "${VGGT_LONG_WEIGHTS_MIN_MB}" ]]; then
    return 1
  fi

  [[ -f "${VGGT_LONG_WEIGHTS_DIR}/dinov2_vitb14_pretrain.pth" ]] || return 1
  [[ -f "${VGGT_LONG_WEIGHTS_DIR}/ORBvoc.txt" ]] || return 1

  if [[ -d "${VGGT_LONG_WEIGHTS_DIR}/Pi3" || -d "${VGGT_LONG_WEIGHTS_DIR}/pi3" || -d "${VGGT_LONG_WEIGHTS_DIR}/map_anything" ]]; then
    return 0
  fi

  return 1
}

download_vggt_long_weights_if_needed() {
  mkdir -p "${VGGT_LONG_WEIGHTS_DIR}"

  if [[ ${FORCE_VGGT_LONG_WEIGHTS} -eq 1 ]]; then
    echo "Forcing VGGT-Long weights download because --force-vggt-long-weights was provided."
  elif vggt_long_weights_present; then
    echo "Skipping VGGT-Long weights download; existing weights look complete in ${VGGT_LONG_WEIGHTS_DIR}."
    return
  fi

  echo "Downloading VGGT-Long weights into ${VGGT_LONG_WEIGHTS_DIR} ..."
  (
    cd "${VGGT_LONG_DIR}"
    bash ./scripts/download_weights.sh
  )
  touch "${VGGT_LONG_WEIGHTS_STAMP}"
}

download_sam2_checkpoint_if_needed() {
  # Fetch the SAM 2.1 Hiera-Large checkpoint from its direct fbaipublicfiles URL
  # (same fully-scripted idiom as scripts/fetch_tapir_checkpoint.sh). Idempotent:
  # skip a plausibly-sized existing file; download to a temp file + mv so an
  # interrupted transfer never leaves a partial that the size guard accepts.
  mkdir -p "${SAM2_CKPT_DIR}"
  local mb
  if [[ -f "${SAM2_CKPT}" ]]; then
    mb="$(du -m "${SAM2_CKPT}" 2>/dev/null | awk '{print $1}')"
    if [[ -n "${mb}" && "${mb}" -ge "${SAM2_CKPT_MIN_MB}" ]]; then
      echo "SAM 2 checkpoint already present (>= ${SAM2_CKPT_MIN_MB} MB): ${SAM2_CKPT} — skipping."
      return
    fi
  fi
  echo "Downloading SAM 2.1 Hiera-Large checkpoint -> ${SAM2_CKPT} ..."
  local tmp
  tmp="$(mktemp "${SAM2_CKPT}.XXXXXX")"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 -o "${tmp}" "${SAM2_CKPT_URL}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${tmp}" "${SAM2_CKPT_URL}"
  else
    rm -f "${tmp}"
    echo "Neither curl nor wget is available to download the SAM 2 checkpoint." >&2
    exit 1
  fi
  mb="$(du -m "${tmp}" 2>/dev/null | awk '{print $1}')"
  if [[ -z "${mb}" || "${mb}" -lt "${SAM2_CKPT_MIN_MB}" ]]; then
    rm -f "${tmp}"
    echo "Downloaded SAM 2 checkpoint smaller than ${SAM2_CKPT_MIN_MB} MB — treating as failed/truncated." >&2
    exit 1
  fi
  mv "${tmp}" "${SAM2_CKPT}"
  echo "Done: ${SAM2_CKPT} (${mb} MB)"
}

download_dinov2_weights_if_needed() {
  # Pre-cache the DINOv2-L weights into the HF hub cache via transformers. This is
  # idempotent (from_pretrained reuses an already-downloaded snapshot) and ungated
  # (DINOv2 is Apache). The gated DINOv3-L swap instead needs `hf auth login` +
  # granted access — same call, different model_id.
  echo "Pre-caching DINOv2 weights (${DINOV2_MODEL_ID}) into the HF hub cache..."
  DINOV2_MODEL_ID="${DINOV2_MODEL_ID}" python - <<'PYEOF'
import os
from transformers import AutoImageProcessor, AutoModel
mid = os.environ["DINOV2_MODEL_ID"]
AutoImageProcessor.from_pretrained(mid)
AutoModel.from_pretrained(mid)
print("DINOv2 weights cached:", mid)
PYEOF
}

download_siglip2_weights_if_needed() {
  # Pre-cache the SigLIP2 weights (image+text encoder) into the HF hub cache via
  # transformers. Idempotent (from_pretrained reuses the snapshot) and ungated
  # (SigLIP2 is Apache) — no `hf auth login` needed, unlike the gated facebook models.
  echo "Pre-caching SigLIP2 weights (${SIGLIP2_MODEL_ID}) into the HF hub cache..."
  SIGLIP2_MODEL_ID="${SIGLIP2_MODEL_ID}" python - <<'PYEOF'
import os
from transformers import AutoModel, AutoProcessor
mid = os.environ["SIGLIP2_MODEL_ID"]
AutoProcessor.from_pretrained(mid)
AutoModel.from_pretrained(mid)
print("SigLIP2 weights cached:", mid)
PYEOF
}

download_sapiens_checkpoints_if_needed() {
  # Pre-cache the Sapiens TorchScript checkpoints (pose/seg/depth/normal) for
  # SAPIENS_SIZE into the HF hub cache. Idempotent + ungated (CC-BY-NC-4.0). Reuses the
  # repo/filename list from the room4d package so it stays the single source of truth.
  echo "Pre-caching Sapiens ${SAPIENS_SIZE} checkpoints (pose/seg/depth/normal) into the HF hub cache..."
  SAPIENS_SIZE="${SAPIENS_SIZE}" python - <<'PYEOF'
import os
from huggingface_hub import hf_hub_download
from room4d.perception.humans.sapiens_wrapper import _CHECKPOINTS
size = os.environ["SAPIENS_SIZE"]
for task, (repo, filename) in _CHECKPOINTS[size].items():
    print(f"  {task}: {hf_hub_download(repo, filename)}")
print("Sapiens checkpoints cached:", size)
PYEOF
}

remove_gsplat_leaked_top_level_packages() {
  # The pinned gsplat wheel ships top-level `tests` and `experimental`
  # packages into site-packages (upstream packaging bug). The leaked `tests`
  # shadows this repo's own tests package and breaks pytest collection, so
  # remove every top-level package gsplat's RECORD claims outside its own
  # namespace. Deleting RECORD-listed files makes a later pip uninstall warn
  # about missing files; that is harmless.
  python - <<'PY'
import shutil
import sysconfig
from importlib.metadata import distribution
from pathlib import Path

site = Path(sysconfig.get_paths()["purelib"])
files = distribution("gsplat").files or []
removed = set()
for f in files:
    top = f.parts[0]
    if top.startswith("gsplat") or top.startswith("."):
        continue
    target = site / top
    if target.exists():
        shutil.rmtree(target) if target.is_dir() else target.unlink()
        removed.add(top)
if removed:
    print(f"Removed gsplat-leaked top-level packages: {sorted(removed)}")
PY
}

install_gsplat() {
  # gsplat is installed with --no-deps below so its torch/numpy declarations
  # cannot fight the locked tree. Its only declared deps not already in the
  # lock are jaxtyping and ninja — install them explicitly, or the env fails
  # `pip check` after every gsplat install. ninja also parallelizes the CUDA
  # source build.
  python -m pip install ninja jaxtyping

  # Idempotency guard: the from-source CUDA build is the single most expensive
  # step in this installer (minutes at MAX_JOBS=4), and pip's wheel cache does
  # not help VCS builds — skip when the pinned COMMIT is already installed.
  # The comparison must be the VCS SHA from pip's direct_url.json, NOT
  # __version__: different gsplat commits report the same version string, so a
  # version check can never correct a wrong-SHA install.
  local installed_sha
  installed_sha="$(python - <<'PY' 2>/dev/null || true
import json
from importlib.metadata import distribution
try:
    dist = distribution("gsplat")
    url_text = dist.read_text("direct_url.json")
    print(json.loads(url_text)["vcs_info"]["commit_id"] if url_text else "")
except Exception:
    print("")
PY
)"
  if [[ "${installed_sha}" == "${GSPLAT_REF}" ]]; then
    echo "gsplat already installed at pinned commit ${GSPLAT_REF:0:12} — skipping source build."
    remove_gsplat_leaked_top_level_packages
    return
  fi
  [[ -n "${installed_sha}" ]] && echo "gsplat installed at ${installed_sha:0:12}, pin is ${GSPLAT_REF:0:12} — rebuilding."
  local spec="git+${GSPLAT_REPO}"
  [[ -n "${GSPLAT_REF}" ]] && spec="${spec}@${GSPLAT_REF}"
  echo "Building gsplat from source for sm_120 (TORCH_CUDA_ARCH_LIST=${GSPLAT_ARCH_LIST})..."
  CUDA_HOME="${CUDA_HOME}" \
  TORCH_CUDA_ARCH_LIST="${GSPLAT_ARCH_LIST}" \
  MAX_JOBS="${MAX_JOBS:-4}" \
    python -m pip install --no-build-isolation --force-reinstall --no-deps "${spec}"
  remove_gsplat_leaked_top_level_packages
}

ensure_ffmpeg() {
  # Install the ffmpeg CLI into the active conda env for dataset prep. Prefer
  # the conda-forge GPL build (build string `gpl_*`) because it bundles
  # libx264; the non-GPL variant omits it and `-crf 0` H.264 would fail. The
  # smoke test below asserts libx264 is actually present.
  if [[ $USE_VENV -eq 1 ]]; then
    echo "Skipping ffmpeg conda install in --use-venv mode; install ffmpeg on PATH manually if you need dataset prep." >&2
    return
  fi
  if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "No active conda environment detected while ensuring ffmpeg." >&2
    exit 1
  fi

  local ffmpeg_bin="${CONDA_PREFIX}/bin/ffmpeg"
  if [[ -x "${ffmpeg_bin}" ]] \
     && "${ffmpeg_bin}" -version 2>/dev/null | head -n 1 | grep -q "ffmpeg version ${FFMPEG_VERSION}" \
     && "${ffmpeg_bin}" -hide_banner -encoders 2>/dev/null | grep -q "libx264"; then
    echo "ffmpeg ${FFMPEG_VERSION} (with libx264) already present at ${ffmpeg_bin} — skipping."
    return
  fi

  echo "Installing ffmpeg ${FFMPEG_VERSION} into env '${CONDA_DEFAULT_ENV:-unknown}' (conda-forge, GPL build with libx264)..."
  if ! run_conda_install_relaxed -y -c conda-forge "ffmpeg=${FFMPEG_VERSION}=gpl*"; then
    echo "Pinned GPL-variant install did not resolve; retrying without the build-string constraint..."
    run_conda_install_relaxed -y -c conda-forge "ffmpeg=${FFMPEG_VERSION}"
  fi
}

ensure_faiss() {
  # faiss for VGGT-Long loop closure (place-recognition retrieval). It is
  # stripped from VGGT-Long's pip requirements (faiss-gpu needs a CUDA build),
  # so install the conda-forge CPU build instead — enough for retrieval, and it
  # avoids a CUDA-version match. Loop closure also needs the dinov2 path patch
  # below; without both, --vggt-long-loop-enable fails.
  if [[ $USE_VENV -eq 1 ]]; then
    echo "Skipping faiss conda install in --use-venv mode; install faiss-cpu manually if you need VGGT-Long loop closure." >&2
    return
  fi
  if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "No active conda environment detected while ensuring faiss; skipping (loop closure will be unavailable)." >&2
    return
  fi
  if python -c "import faiss" >/dev/null 2>&1; then
    echo "faiss already importable — skipping."
    return
  fi
  echo "Installing faiss-cpu ${FAISS_VERSION} into env '${CONDA_DEFAULT_ENV:-unknown}' (conda-forge) for VGGT-Long loop closure..."
  if ! run_conda_install_relaxed -y -c conda-forge "faiss-cpu=${FAISS_VERSION}"; then
    echo "Pinned faiss-cpu install did not resolve; retrying without the version constraint..."
    run_conda_install_relaxed -y -c conda-forge faiss-cpu
  fi
}

ensure_modern_conda_compilers() {
  if [[ $USE_VENV -eq 1 ]]; then
    local current_compiler="${CXX:-c++}"
    local current_major
    current_major="$(compiler_major_version "${current_compiler}")"

    if [[ -z "${current_major}" || "${current_major}" -lt "${REQUIRED_GXX_MAJOR}" ]]; then
      echo "PyTorch CUDA-extension builds require GCC/G++ ${REQUIRED_GXX_MAJOR}+." >&2
      echo "Current compiler: ${current_compiler}" >&2
      echo "Detected version: $("${current_compiler}" --version 2>/dev/null | head -n 1 || echo unknown)" >&2
      echo "When using --use-venv, please provide a newer system compiler manually." >&2
      exit 1
    fi

    echo "Using existing compiler for venv mode: ${current_compiler}"
    return
  fi

  if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "No active conda environment detected while ensuring modern compilers." >&2
    exit 1
  fi

  local conda_cxx="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++"
  local conda_cc="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-cc"

  local current_compiler="${CXX:-c++}"
  local current_major
  current_major="$(compiler_major_version "${current_compiler}")"

  # If conda compilers already exist, prefer them for reproducibility.
  if [[ -x "${conda_cxx}" && -x "${conda_cc}" ]]; then
    export CC="${conda_cc}"
    export CXX="${conda_cxx}"
    echo "Using conda compilers:"
    echo "  CC=${CC}"
    echo "  CXX=${CXX}"
    echo "  CXX version: $("${CXX}" --version | head -n 1)"
    return
  fi

  if [[ -n "${current_major}" && "${current_major}" -ge "${REQUIRED_GXX_MAJOR}" ]]; then
    echo "Existing compiler is already new enough:"
    echo "  ${current_compiler} ($( "${current_compiler}" --version | head -n 1 ))"
    return
  fi

  echo "Current compiler is too old for PyTorch CUDA-extension builds."
  echo "Installing newer conda compilers into env '${CONDA_DEFAULT_ENV:-unknown}'..."
  run_conda_install_relaxed -y -c conda-forge \
    "gcc_linux-64=${CONDA_GCC_VERSION}" \
    "gxx_linux-64=${CONDA_GCC_VERSION}"

  if [[ ! -x "${conda_cxx}" || ! -x "${conda_cc}" ]]; then
    echo "Expected conda compilers were not found after installation." >&2
    echo "Expected:"
    echo "  ${conda_cc}"
    echo "  ${conda_cxx}"
    exit 1
  fi

  export CC="${conda_cc}"
  export CXX="${conda_cxx}"

  echo "Installed and selected conda compilers:"
  echo "  CC=${CC}"
  echo "  CXX=${CXX}"
  echo "  CXX version: $("${CXX}" --version | head -n 1)"
}

activate_conda_env() {
  local target_env="$1"

  if ! command -v conda >/dev/null 2>&1 && [[ -z "${CONDA_EXE:-}" ]]; then
    echo "Conda is required unless --use-venv is provided, but 'conda' was not found." >&2
    exit 1
  fi

  local conda_base=""
  if [[ -n "${CONDA_EXE:-}" ]]; then
    conda_base="$(dirname "$(dirname "${CONDA_EXE}")")"
  else
    conda_base="$(conda info --base)"
  fi

  # shellcheck disable=SC1091
  source "${conda_base}/etc/profile.d/conda.sh"

  if [[ -n "${target_env}" ]]; then
    if [[ "${CONDA_DEFAULT_ENV:-}" == "${target_env}" ]]; then
      echo "Using already-active conda environment: ${target_env}"
      return
    fi

    echo "Activating conda environment: ${target_env}"
    if ! conda activate "${target_env}"; then
      echo "Failed to activate conda environment '${target_env}'." >&2
      exit 1
    fi
    return
  fi

  if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "Using already-active conda environment: ${CONDA_DEFAULT_ENV}"
    return
  fi

  echo "No active conda environment detected. Provide --conda-env NAME or use --use-venv." >&2
  exit 1
}

setup_python_env() {
  if [[ $USE_VENV -eq 1 ]]; then
    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
      echo "Python executable not found: $PYTHON_BIN" >&2
      exit 1
    fi

    echo "Creating / using repo-local venv: ${VENV_DIR}"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
  else
    activate_conda_env "${CONDA_ENV_NAME}"
  fi

  echo "Python executable: $(command -v python)"
  python -m pip install --upgrade pip setuptools wheel
}

install_vggt_long_requirements() {
  local req_file="${VGGT_LONG_DIR}/requirements.txt"

  if [[ ! -f "${req_file}" ]]; then
    return
  fi

  echo "Installing VGGT-Long requirements from ${req_file}..."

  local tmp_req
  tmp_req="$(mktemp)"

  # Remove packages we intentionally do not install automatically here:
  # - xformers: requires building against a local CUDA toolkit matching torch
  # - faiss-gpu: should be installed via conda if needed
  grep -Eiv '^[[:space:]]*(xformers|faiss-gpu)([[:space:]]*(==|=|>=|<=|!=|~=).*)?[[:space:]]*$' "${req_file}" > "${tmp_req}" || true

  if [[ -s "${tmp_req}" ]]; then
    python -m pip install -r "${tmp_req}"
  fi
  rm -f "${tmp_req}"

  if grep -Eqi '^[[:space:]]*xformers([[:space:]]*(==|=|>=|<=|!=|~=).*)?[[:space:]]*$' "${req_file}"; then
    local torch_cuda_version=""
    local local_nvcc_version=""
    torch_cuda_version="$(detect_torch_cuda_version | tr -d '[:space:]')"
    local_nvcc_version="$(nvcc_cuda_version | tr -d '[:space:]')"
    echo "Skipping xformers automatic install."
    echo "Reason: xformers often needs a local CUDA toolkit/build chain aligned with torch."
    echo "Current torch CUDA: ${torch_cuda_version:-unknown}; local nvcc CUDA: ${local_nvcc_version:-not detected}."
    echo "Install xformers manually later only if VGGT-Long actually needs it."
  fi

  if grep -Eqi '^[[:space:]]*faiss-gpu([[:space:]]*(==|=|>=|<=|!=|~=).*)?[[:space:]]*$' "${req_file}"; then
    echo "Skipping faiss-gpu from pip requirements."
    echo "The conda-forge faiss-cpu build is installed separately by ensure_faiss (for loop closure)."
  fi
}

smoke_test_python_module() {
  local label="$1"
  local module_name="$2"
  local dist_name="${3:-$2}"
  local extra_path="${4:-}"

  python - "$label" "$module_name" "$dist_name" "$extra_path" <<'PYCODE'
import importlib
import importlib.metadata as metadata
import sys

label, module_name, dist_name, extra_path = sys.argv[1:5]
if extra_path:
    sys.path.insert(0, extra_path)
module = importlib.import_module(module_name)
try:
    version = metadata.version(dist_name)
except Exception:
    version = getattr(module, '__version__', 'unknown')
location = getattr(module, '__file__', None)
if location is None:
    path = getattr(module, '__path__', None)
    if path:
        try:
            location = next(iter(path))
        except Exception:
            location = '<namespace>'
print(f"[smoke-test] OK {label}: module={module_name} version={version} path={location or '<unknown>'}")
PYCODE
}

smoke_test_gsplat_rasterization() {
  # An import-only check would pass a broken build — rasterize a trivial scene
  # so the sm_120 CUDA kernels the source build exists for actually execute.
  python - <<'PYCODE'
import torch
import gsplat

assert torch.cuda.is_available(), "gsplat smoke test needs a CUDA device"
n = 64
means = torch.randn(n, 3, device="cuda")
quats = torch.randn(n, 4, device="cuda")
quats = quats / quats.norm(dim=-1, keepdim=True)
scales = torch.rand(n, 3, device="cuda") * 0.1
opacities = torch.rand(n, device="cuda")
colors = torch.rand(n, 3, device="cuda")
viewmat = torch.eye(4, device="cuda")[None]
K = torch.tensor([[[100.0, 0, 64], [0, 100.0, 64], [0, 0, 1]]], device="cuda")
rgb, alpha, _ = gsplat.rasterization(
    means, quats, scales, opacities, colors, viewmat, K, 128, 128
)
assert rgb.shape == (1, 128, 128, 3), f"unexpected output shape {tuple(rgb.shape)}"
version = getattr(gsplat, "__version__", "unknown")
print(f"[smoke-test] OK gsplat rasterization: version={version} output={tuple(rgb.shape)}")
PYCODE
}

smoke_test_ffmpeg() {
  local ffmpeg_bin="${CONDA_PREFIX:-}/bin/ffmpeg"
  if [[ ! -x "${ffmpeg_bin}" ]]; then
    ffmpeg_bin="$(command -v ffmpeg || true)"
  fi
  if [[ -z "${ffmpeg_bin}" || ! -x "${ffmpeg_bin}" ]]; then
    echo "[smoke-test] FAIL ffmpeg: binary not found after install" >&2
    exit 1
  fi
  if ! "${ffmpeg_bin}" -hide_banner -encoders 2>/dev/null | grep -q "libx264"; then
    echo "[smoke-test] FAIL ffmpeg: libx264 encoder missing (need the GPL build for -crf 0 H.264)" >&2
    exit 1
  fi
  local ver
  ver="$("${ffmpeg_bin}" -version 2>/dev/null | head -n 1)"
  echo "[smoke-test] OK ffmpeg: ${ver} path=${ffmpeg_bin} (libx264 present)"
}

run_post_install_smoke_tests() {
  echo
  echo "Running post-install smoke tests..."
  smoke_test_python_module "room4d" "room4d" "room4d"

  if [[ $INSTALL_FFMPEG -eq 1 ]]; then
    smoke_test_ffmpeg
  fi

  if [[ $INSTALL_DETECTRON2 -eq 1 ]]; then
    smoke_test_python_module "detectron2" "detectron2" "detectron2"
  fi
  if [[ $INSTALL_VGGT -eq 1 ]]; then
    smoke_test_python_module "VGGT" "vggt" "vggt"
  fi
  if [[ $INSTALL_COTRACKER -eq 1 ]]; then
    smoke_test_python_module "CoTracker" "cotracker" "cotracker"
  fi
  if [[ $INSTALL_SAM2 -eq 1 ]]; then
    smoke_test_python_module "SAM2" "sam2" "SAM-2"
  fi
  if [[ $INSTALL_DINOV2 -eq 1 ]]; then
    smoke_test_python_module "transformers" "transformers" "transformers"
  fi
  if [[ $INSTALL_SIGLIP2 -eq 1 ]]; then
    smoke_test_python_module "transformers" "transformers" "transformers"
  fi
  if [[ $INSTALL_SAPIENS -eq 1 ]]; then
    smoke_test_python_module "torch" "torch" "torch"
  fi
  if [[ $INSTALL_POSE_REFINE -eq 1 ]]; then
    # pose-refinement deps: pycolmap (BA), hydra-core + omegaconf (required by
    # the vendored VGGSfM tracker the refiner drives).
    smoke_test_python_module "pycolmap" "pycolmap" "pycolmap"
    smoke_test_python_module "hydra-core" "hydra" "hydra-core"
    smoke_test_python_module "omegaconf" "omegaconf" "omegaconf"
  fi
  if [[ $INSTALL_CALIBRATION -eq 1 ]]; then
    smoke_test_python_module "pycolmap" "pycolmap" "pycolmap"
    smoke_test_python_module "hloc" "hloc" "hloc"
    # The hloc patches above set specific fields on pycolmap option objects;
    # a pycolmap bump that renames them would otherwise only fail (or
    # silently no-op) at the first real calibration run.
    echo "Smoke-testing pycolmap option fields used by the hloc patches..."
    python - <<'PYEOF'
import pycolmap
ro = pycolmap.ImageReaderOptions()
ro.camera_model = "SIMPLE_PINHOLE"
ro.camera_params = "500,320,240"
tv = pycolmap.TwoViewGeometryOptions()
tv.ransac.max_num_trials = 20000
tv.ransac.min_inlier_ratio = 0.1
ip = pycolmap.IncrementalPipelineOptions()
ip.mergedict({"ba_refine_focal_length": False, "ba_refine_extra_params": False})
print("pycolmap option fields OK")
PYEOF
  fi
  if [[ $INSTALL_VGGT_LONG -eq 1 ]]; then
    smoke_test_python_module "VGGT-Long" "vggt_long" "vggt-long" "${VGGT_LONG_DIR}"
  fi
  if [[ $INSTALL_GSPLAT -eq 1 ]]; then
    smoke_test_gsplat_rasterization
  fi
}

setup_python_env

# Install the locked pip tree first (default when the lock exists): the
# editable room4d install below then finds every dependency already satisfied
# at the exact known-good version instead of re-resolving ranges.
if [[ $UNLOCKED -eq 0 && -f "${ROOM4D_LOCK_FILE}" ]]; then
  echo "Installing pinned dependency tree from ${ROOM4D_LOCK_FILE} (use --unlocked to re-resolve)..."
  python -m pip install -r "${ROOM4D_LOCK_FILE}" \
    --extra-index-url "${TORCH_INDEX_URL}" \
    --extra-index-url "https://pypi.nvidia.com"
elif [[ $UNLOCKED -eq 0 ]]; then
  echo "No lockfile at ${ROOM4D_LOCK_FILE} — falling back to fresh resolution."
fi

EXTRAS=()
[[ $INSTALL_VISION -eq 1 ]] && EXTRAS+=(vision)
[[ $INSTALL_GEOMETRY -eq 1 ]] && EXTRAS+=(geometry)
[[ $INSTALL_CALIBRATION -eq 1 ]] && EXTRAS+=(calibration)
[[ $INSTALL_POSE_REFINE -eq 1 ]] && EXTRAS+=(pose_refine)
[[ $INSTALL_VGGT_LONG -eq 1 ]] && EXTRAS+=(vggt_long)
[[ $INSTALL_DEV -eq 1 ]] && EXTRAS+=(dev)

# Install room4d and all Python dependencies from pyproject.toml only.
if [[ ${#EXTRAS[@]} -gt 0 ]]; then
  EXTRA_CSV=$(IFS=,; echo "${EXTRAS[*]}")
  python -m pip install -e "${ROOT_DIR}[${EXTRA_CSV}]"
else
  python -m pip install -e "${ROOT_DIR}"
fi

if [[ $INSTALL_DETECTRON2 -eq 1 ]]; then
  TMP_DIR=$(mktemp -d)
  echo "Installing detectron2 from official tag ${DETECTRON2_TAG}..."
  git clone --depth 1 --branch "${DETECTRON2_TAG}" --single-branch "${DETECTRON2_REPO}" "$TMP_DIR/detectron2"
  python -m pip install --no-build-isolation -e "$TMP_DIR/detectron2"
fi

if [[ $INSTALL_VGGT -eq 1 ]]; then
  clone_or_update_checkout "${VGGT_REPO}" "${VGGT_DIR}" "${VGGT_COMMIT}"
  echo "Installing VGGT from ${VGGT_DIR}..."
  python -m pip install -e "${VGGT_DIR}"
fi

if [[ $INSTALL_COTRACKER -eq 1 ]]; then
  clone_or_update_checkout "${COTRACKER_REPO}" "${COTRACKER_DIR}" "${COTRACKER_COMMIT}"
  echo "Installing CoTracker from ${COTRACKER_DIR}..."
  python -m pip install -e "${COTRACKER_DIR}"
fi

if [[ $INSTALL_SAM2 -eq 1 ]]; then
  clone_or_update_checkout "${SAM2_REPO}" "${SAM2_DIR}" "${SAM2_COMMIT}"
  echo "Installing SAM 2 from ${SAM2_DIR} (SAM2_BUILD_CUDA=0 — optional ext skipped)..."
  # The connected-components CUDA ext is optional (Python fallback exists);
  # skipping it keeps the install off the CUDA-toolkit/compiler path.
  SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation -e "${SAM2_DIR}"
  download_sam2_checkpoint_if_needed
fi

if [[ $INSTALL_DINOV2 -eq 1 ]]; then
  echo "Installing HuggingFace transformers (${TRANSFORMERS_SPEC}) for the DINOv2 reid rung..."
  python -m pip install "${TRANSFORMERS_SPEC}"
  download_dinov2_weights_if_needed
fi

if [[ $INSTALL_SIGLIP2 -eq 1 ]]; then
  echo "Installing HuggingFace transformers (${TRANSFORMERS_SPEC}) for the SigLIP2 semantics rung..."
  python -m pip install "${TRANSFORMERS_SPEC}"
  download_siglip2_weights_if_needed
fi

if [[ $INSTALL_SAPIENS -eq 1 ]]; then
  download_sapiens_checkpoints_if_needed
fi

if [[ $INSTALL_CALIBRATION -eq 1 ]]; then
  clone_or_update_checkout "${HLOC_REPO}" "${HLOC_DIR}" "${HLOC_TAG}"
  if [[ -f "${HLOC_DIR}/.gitmodules" ]]; then
    git -C "${HLOC_DIR}" submodule update --init --recursive
  fi
  echo "Installing hloc from ${HLOC_DIR} (tag ${HLOC_TAG})..."
  python -m pip install -e "${HLOC_DIR}"

  # Patch hloc for pycolmap >= 3.10 API drift (the pinned hloc tag predates
  # it): import_images renamed image_list->image_names and requires an
  # ImageReaderOptions instance; verify_matches moved its RANSAC kwargs into
  # TwoViewGeometryOptions; incremental_mapping requires
  # IncrementalPipelineOptions; all three require str paths. Idempotent:
  # each replacement is skipped when already applied, and a missing anchor
  # fails LOUDLY instead of silently shipping an unpatched tree.
  echo "Patching hloc for pycolmap >= 3.10 API drift..."
  python - "${HLOC_DIR}" <<'PYEOF'
import sys
from pathlib import Path

hloc_dir = Path(sys.argv[1])
PATCHES = [
    (hloc_dir / "hloc" / "reconstruction.py", """    with pycolmap.ostream():
        pycolmap.import_images(database_path, image_dir, camera_mode,
                               image_list=image_list or [],
                               options=options)""", """    with pycolmap.ostream():
        # room4d patch (applied by scripts/setup_room4d.sh): pycolmap >= 3.10
        # renamed the image_list kwarg to image_names and requires an
        # ImageReaderOptions instance (dicts are no longer accepted); paths
        # must be str.
        reader_options = pycolmap.ImageReaderOptions()
        for key, value in (options or {}).items():
            setattr(reader_options, key, value)
        pycolmap.import_images(str(database_path), str(image_dir), camera_mode,
                               image_names=image_list or [],
                               options=reader_options)"""),
    (hloc_dir / "hloc" / "reconstruction.py", """            reconstructions = pycolmap.incremental_mapping(
                database_path, image_dir, models_path, options=options)""", """            # room4d patch (applied by scripts/setup_room4d.sh): pycolmap
            # >= 3.10 requires IncrementalPipelineOptions and str paths.
            mapper_options_obj = pycolmap.IncrementalPipelineOptions()
            if options:
                mapper_options_obj.mergedict(options)
            reconstructions = pycolmap.incremental_mapping(
                str(database_path), str(image_dir), str(models_path),
                options=mapper_options_obj)"""),
    (hloc_dir / "hloc" / "triangulation.py", """    with OutputCapture(verbose):
        with pycolmap.ostream():
            pycolmap.verify_matches(
                database_path, pairs_path,
                max_num_trials=20000, min_inlier_ratio=0.1)""", """    with OutputCapture(verbose):
        with pycolmap.ostream():
            # room4d patch (applied by scripts/setup_room4d.sh): pycolmap
            # >= 3.10 moved the RANSAC kwargs into TwoViewGeometryOptions
            # and requires str paths.
            options = pycolmap.TwoViewGeometryOptions()
            options.ransac.max_num_trials = 20000
            options.ransac.min_inlier_ratio = 0.1
            pycolmap.verify_matches(str(database_path), str(pairs_path),
                                    options=options)"""),
]
for path, old, new in PATCHES:
    src = path.read_text()
    if new in src:
        continue  # already patched
    if old not in src:
        sys.exit(f"hloc patch anchor not found in {path} -- upstream "
                 "drifted; update the patch in setup_room4d.sh")
    path.write_text(src.replace(old, new))
    print(f"  patched {path.name}")
PYEOF
fi

if [[ $INSTALL_VGGT_LONG -eq 1 ]]; then
  ensure_modern_conda_compilers
  clone_or_update_checkout "${VGGT_LONG_REPO}" "${VGGT_LONG_DIR}" "${VGGT_LONG_COMMIT}"
  install_vggt_long_requirements
  ensure_faiss
  ensure_cuda_toolkit_for_torch_extensions

  # Patch VGGT-Long's loop-closure DINOv2 backbone: it torch.hub.load's
  # './LoopModels/dinov2' relative to the process CWD, so loop closure breaks
  # whenever VGGT-Long runs from any directory but its own (always, under
  # room4d). Anchor the path to the module's own location. Idempotent; fails
  # loudly if the upstream anchor is gone.
  echo "Patching VGGT-Long loop-closure DINOv2 path (CWD-relative -> __file__-anchored)..."
  python - "${VGGT_LONG_DIR}" <<'PYEOF'
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "LoopModels" / "backbones" / "dinov2.py"
OLD = "        self.model = torch.hub.load('./LoopModels/dinov2', model_name, source='local', pretrained=False)"
NEW = (
    "        # room4d patch (applied by scripts/setup_room4d.sh): the upstream './LoopModels/dinov2'\n"
    "        # path is resolved against the process CWD, so loop closure breaks whenever VGGT-Long\n"
    "        # runs from any directory but its own (always, under room4d). Anchor it to this file.\n"
    "        import os as _os\n"
    "        _dinov2_repo = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'dinov2')\n"
    "        self.model = torch.hub.load(_dinov2_repo, model_name, source='local', pretrained=False)"
)
src = path.read_text()
if "_dinov2_repo" in src:
    print("  dinov2 path already patched")
elif OLD in src:
    path.write_text(src.replace(OLD, NEW))
    print(f"  patched {path.name}")
else:
    sys.exit(f"dinov2 patch anchor not found in {path} -- upstream drifted; update the patch in setup_room4d.sh")
PYEOF

  if [[ -f "${VGGT_LONG_DIR}/scripts/download_weights.sh" ]]; then
    echo "Patching VGGT-Long download_weights.sh for current Hugging Face CLI..."
    sed -i 's/\bhuggingface-cli\b/hf/g' "${VGGT_LONG_DIR}/scripts/download_weights.sh"
    sed -i 's/[[:space:]]*--local-dir-use-symlinks[[:space:]]\+False//g' "${VGGT_LONG_DIR}/scripts/download_weights.sh"
  fi

  # Provide a compatibility shim if the environment has `hf` but not `huggingface-cli`.
  if ! command -v huggingface-cli >/dev/null 2>&1 && command -v hf >/dev/null 2>&1; then
    mkdir -p "${CONDA_PREFIX:-${ROOT_DIR}/.room4d_cache}/bin"
    cat > "${CONDA_PREFIX:-${ROOT_DIR}/.room4d_cache}/bin/huggingface-cli" <<'EOF'
#!/usr/bin/env bash
exec hf "$@"
EOF
    chmod +x "${CONDA_PREFIX:-${ROOT_DIR}/.room4d_cache}/bin/huggingface-cli"
    export PATH="${CONDA_PREFIX:-${ROOT_DIR}/.room4d_cache}/bin:${PATH}"
  fi

  download_vggt_long_weights_if_needed

  echo "Installing VGGT-Long from ${VGGT_LONG_DIR}..."
  MAX_JOBS="${MAX_JOBS:-4}" python -m pip install --no-build-isolation -e "${VGGT_LONG_DIR}"
fi

if [[ $INSTALL_GSPLAT -eq 1 ]]; then
  ensure_modern_conda_compilers
  ensure_cuda_toolkit_for_torch_extensions
  install_gsplat
fi

if [[ $INSTALL_FFMPEG -eq 1 ]]; then
  ensure_ffmpeg
fi

run_post_install_smoke_tests

cat <<EOF

Setup complete.

Environment:
$( [[ $USE_VENV -eq 1 ]] && echo "- repo-local venv: ${VENV_DIR}" || echo "- conda env: ${CONDA_DEFAULT_ENV:-unknown}" )

Installed:
- room4d in editable mode from:
    ${ROOT_DIR}
- Python dependencies from pyproject.toml
$( [[ $INSTALL_DETECTRON2 -eq 1 ]] && echo "- detectron2 from ${DETECTRON2_TAG}" )
$( [[ $INSTALL_VGGT -eq 1 ]] && echo "- VGGT from commit ${VGGT_COMMIT} at ${VGGT_DIR}" )
$( [[ $INSTALL_COTRACKER -eq 1 ]] && echo "- CoTracker from commit ${COTRACKER_COMMIT} at ${COTRACKER_DIR}" )
$( [[ $INSTALL_SAM2 -eq 1 ]] && echo "- SAM 2 from commit ${SAM2_COMMIT} at ${SAM2_DIR} (+ sam2.1_hiera_large checkpoint)" )
$( [[ $INSTALL_DINOV2 -eq 1 ]] && echo "- transformers (${TRANSFORMERS_SPEC}) + DINOv2-L weights ${DINOV2_MODEL_ID} cached (FP3 reid rung)" )
$( [[ $INSTALL_SIGLIP2 -eq 1 ]] && echo "- transformers (${TRANSFORMERS_SPEC}) + SigLIP2 weights ${SIGLIP2_MODEL_ID} cached (FP4 semantics rung)" )
$( [[ $INSTALL_SAPIENS -eq 1 ]] && echo "- Sapiens ${SAPIENS_SIZE} dense checkpoints cached (FP5 humans rung; CC-BY-NC-4.0)" )
$( [[ $INSTALL_CALIBRATION -eq 1 ]] && echo "- calibration extras from pyproject.toml (pycolmap ${PYCOLMAP_VERSION}, matching COLMAP ${COLMAP_UPSTREAM_TAG})" )
$( [[ $INSTALL_CALIBRATION -eq 1 ]] && echo "- hloc from tag ${HLOC_TAG} at ${HLOC_DIR}" )
$( [[ $INSTALL_VGGT_LONG -eq 1 ]] && echo "- VGGT-Long from commit ${VGGT_LONG_COMMIT} at ${VGGT_LONG_DIR}" )
$( [[ $INSTALL_VGGT_LONG -eq 1 ]] && echo "- VGGT-Long loop closure enabled: faiss-cpu ${FAISS_VERSION} + DINOv2 path patch (CWD-relative -> __file__)" )
$( [[ $INSTALL_GSPLAT -eq 1 ]] && echo "- gsplat built from source for sm_120 at commit ${GSPLAT_REF}" )
$( [[ $INSTALL_FFMPEG -eq 1 ]] && echo "- ffmpeg ${FFMPEG_VERSION} (conda-forge GPL build with libx264)" )

Notes:
- third_party/ should be gitignored so external source checkouts are never tracked.
- .room4d_cache/ should be gitignored too.
- Calibration mode is pinned to hloc ${HLOC_TAG} and pycolmap ${PYCOLMAP_VERSION}
  (aligned with upstream COLMAP ${COLMAP_UPSTREAM_TAG}).
- Post-install smoke tests verify the third-party repos and pinned Python packages
  this script installs directly into the active environment.
- For Intel RealSense SDK / pyrealsense2 on Ubuntu, see:
    scripts/setup_realsense_sdk.sh
- For a first RGB-only camera test without RealSense SDK, you can use:
    python scripts/capture_uvc_video.py --device 0 --output out.mp4
EOF
