#!/usr/bin/env bash
set -euo pipefail

# Reproducible installer for the current reifyscene prototype.
#
# What this installs today:
# - the local reifyscene package in editable mode
# - Python dependencies declared in pyproject.toml
# - optional detectron2 from its official v0.6 release tag
# - optional VGGT from the official repo at a pinned commit, cloned into
#   third_party/vggt and installed into the active environment
# - optional CoTracker from the official repo at a pinned commit, cloned into
#   third_party/co-tracker and installed into the active environment
# - optional VGGT-Long from the official repo at a pinned commit, cloned into
#   third_party/VGGT-Long and installed into the active environment
# - optional VGGT-Omega from the official repo at a pinned commit, cloned into
#   third_party/vggt-omega and installed (without its dependencies, which the lock
#   already pins); its weights are gated and download on first use
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
# Pinned to a main commit, NOT the v0.6 tag (Nov 2021): v0.6 pins black==21.4b2,
# which requires pathspec<1 and therefore cannot coexist with a modern mypy — it
# made the generated lockfile unresolvable. On main, `black` is unpinned in
# install_requires (black==22.3.0 lives only in the dev extra).
DETECTRON2_COMMIT="a2f4a8771ab77e8411c26b27f24f9489a28a2453"  # main @ 2026-08-19

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
# Upper bound for the bootstrap setuptools install. Must satisfy the pinned
# torch (2.11.0+cu130 declares setuptools<82); bump alongside a torch pin bump.
SETUPTOOLS_MAX_SPEC="${SETUPTOOLS_MAX_SPEC:-<82}"

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
# filename list lives in reifyscene.perception.humans.sapiens_wrapper._CHECKPOINTS.
SAPIENS_SIZE="${SAPIENS_SIZE:-0.6b}"

# MoGe-2 (Microsoft, MIT, ungated) — the FP6 `geometry` depth backstop rung: a
# single-image metric point map (+ depth / normals / intrinsics) that fills the
# GeometryArtifact depth/pointmap contract below VGGT on the depth ladder
# (VGGT -> MoGe-2 -> Metric3D/UniDepth -> triangulate -> ground-plane, design
# §4A). Standalone package (not transformers): cloned into third_party/MoGe and
# installed with --no-deps (its unpinned torchvision + GUI-only gradio must not
# fight the locked +cu130 tree — same guard as gsplat), then its real inference-path
# deps added explicitly so --with-moge is self-contained under any flag combo:
# utils3d (pinned in MOGE_UTILS3D_SPEC to the commit MoGe requires — co-versioned
# with MOGE_COMMIT), opencv-python (cv2), and scipy. matplotlib is vis-only, OFF the
# infer path (verified by tracing the import + a smoke infer). Then the weights are
# pre-cached AND smoke-inferred from the HF hub (~1.3 GB, ungated). Model id:
# MOGE_MODEL_ID, which the FP6 wrapper (reifyscene.geometry.moge_wrapper, once it lands)
# reads as its default (override to swap to a lighter vit-b/-s variant).
MOGE_MODEL_ID="${MOGE_MODEL_ID:-Ruicheng/moge-2-vitl-normal}"
MOGE_UTILS3D_SPEC="${MOGE_UTILS3D_SPEC:-utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183}"
MOGE_REPO="https://github.com/microsoft/MoGe.git"
MOGE_COMMIT="07444410f1e33f402353b99d6ccd26bd31e469e8"

# VGGT-Omega (FAIR Noncommercial; weights gated on Hugging Face, facebook/VGGT-Omega):
# the one-pass geometry backend (--geometry-backend-name vggt-omega). Its declared
# dependencies (numpy<2, Pillow, einops, safetensors, opencv-python) are all in the
# lock, so it installs with --no-deps.
VGGT_OMEGA_REPO="https://github.com/facebookresearch/vggt-omega.git"
VGGT_OMEGA_COMMIT="48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6"

VGGT_LONG_REPO="https://github.com/DengKaiCQ/VGGT-Long.git"
VGGT_LONG_COMMIT="c160869d1d99c96bb227f414afb3bc68c29c9a76"

# VLA-3D's scene-graph relation functions (MIT): the graph stage's relation computer calls them
# (reifyscene/graph/relations.py). Two patches in third_party_patches/vla-3d/.
VLA3D_REPO="https://github.com/HaochenZ11/VLA-3D.git"
VLA3D_COMMIT="a0c023c9662bd875f0b3aeff4ff60adc1991e449"

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
# DAVIS) into the mp4s the pipeline ingests. reifyscene's own decode path uses
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
CONDA_ENV_NAME="${CONDA_ENV_NAME:-reifyscene}"
CREATE_ENV=0
USE_VENV=0
UNLOCKED=0

# Full pip-layer lockfile (see scripts/gen_env_locks.sh). Installed before the
# editable reifyscene install so the transitive dependency tree replays the
# known-good snapshot instead of re-resolving. torch +cu130 wheels only exist
# on the pytorch index, so the lock install needs it on a fresh env.
REIFYSCENE_LOCK_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/locks/reifyscene.lock"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"

INSTALL_DETECTRON2=0
INSTALL_VGGT=0
INSTALL_COTRACKER=0
INSTALL_SAM2=0
INSTALL_DINOV2=0
INSTALL_SIGLIP2=0
INSTALL_SAPIENS=0
INSTALL_MOGE=0
INSTALL_VGGT_LONG=0
INSTALL_VGGT_OMEGA=0
INSTALL_VLA3D=1
INSTALL_GSPLAT=1
INSTALL_DEV=0
INSTALL_VISION=1
INSTALL_GEOMETRY=1
INSTALL_CALIBRATION=1
INSTALL_POSE_REFINE=1
INSTALL_FFMPEG=1
FORCE_VGGT_LONG_WEIGHTS=0

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Every reifyscene edit to third-party code goes through third_party_patches/ + this helper.
# shellcheck source=scripts/lib/patch_series.sh
source "${ROOT_DIR}/scripts/lib/patch_series.sh"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"
VGGT_DIR="${THIRD_PARTY_DIR}/vggt"
COTRACKER_DIR="${THIRD_PARTY_DIR}/co-tracker"
SAM2_DIR="${THIRD_PARTY_DIR}/sam2"
SAM2_CKPT_DIR="${SAM2_DIR}/checkpoints"
SAM2_CKPT="${SAM2_CKPT_DIR}/sam2.1_hiera_large.pt"
VGGT_LONG_DIR="${THIRD_PARTY_DIR}/VGGT-Long"
VGGT_OMEGA_DIR="${THIRD_PARTY_DIR}/vggt-omega"
MOGE_DIR="${THIRD_PARTY_DIR}/MoGe"
HLOC_DIR="${THIRD_PARTY_DIR}/Hierarchical-Localization"
VLA3D_DIR="${THIRD_PARTY_DIR}/VLA-3D"
VGGT_LONG_WEIGHTS_DIR="${VGGT_LONG_DIR}/weights"
VGGT_LONG_WEIGHTS_STAMP="${VGGT_LONG_WEIGHTS_DIR}/.reifyscene_download_complete"
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
    --create-env)
      CREATE_ENV=1; shift ;;
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
    --with-moge)
      INSTALL_MOGE=1; shift ;;
    --with-vggt-long)
      INSTALL_VGGT_LONG=1; shift ;;
    --with-vggt-omega)
      INSTALL_VGGT_OMEGA=1; shift ;;
    --no-vla3d)
      INSTALL_VLA3D=0; shift ;;
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
Usage: scripts/setup_reifyscene.sh [options]

Environment options:
  --conda-env NAME        Require / activate this conda environment
                          (default: reifyscene)
  --create-env            Create the conda env if it does not exist. Without
                          this flag a missing env is an error, so a typo in
                          --conda-env fails loudly instead of silently building
                          a multi-GB environment. Python version is read from
                          the lockfile header.
  --use-venv              Explicitly use a repo-local venv instead of conda
  --python PATH           Python executable to use (default: python3)
  --venv PATH             Virtualenv path (default: .venv)

Install options:
  --with-detectron2       Install detectron2 (pinned main commit) for DensePose.
                          OFF by default and not yet consumed by any ladder rung.
                          Intended as the permissively-licensed dense top rung,
                          for when Sapiens is ruled out by its CC-BY-NC-4.0
                          licence, its gated weights, or its size. Downgrades
                          iopath below what SAM-2 declares — see the install site.
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
  --with-moge             Clone/install MoGe-2 into third_party/MoGe at pinned commit
                          (--no-deps + utils3d) + pre-cache weights (FP6 geometry depth
                          backstop; MIT, ungated, single-image metric point map)
  --with-vggt-long        Clone/install VGGT-Long into third_party/VGGT-Long at pinned commit
  --with-vggt-omega       Clone/install VGGT-Omega into third_party/vggt-omega at pinned commit
                          (its weights are gated: request access to facebook/VGGT-Omega on
                          Hugging Face and run `hf auth login`; they download on first use)
  --with-gsplat           Build gsplat from source for sm_120 at pinned commit
                          (default: on — the reifyscene renderer requires it)
  --no-vla3d              Skip VLA-3D (scene-graph relations; on by default)
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
                          pinned tree from scripts/locks/reifyscene.lock
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
  # repo/filename list from the reifyscene package so it stays the single source of truth.
  echo "Pre-caching Sapiens ${SAPIENS_SIZE} checkpoints (pose/seg/depth/normal) into the HF hub cache..."
  SAPIENS_SIZE="${SAPIENS_SIZE}" python - <<'PYEOF'
import os
from huggingface_hub import hf_hub_download
from reifyscene.perception.humans.sapiens_wrapper import _CHECKPOINTS
size = os.environ["SAPIENS_SIZE"]
for task, (repo, filename) in _CHECKPOINTS[size].items():
    print(f"  {task}: {hf_hub_download(repo, filename)}")
print("Sapiens checkpoints cached:", size)
PYEOF
}

download_moge_weights_if_needed() {
  # Pre-cache the MoGe-2 weights into the HF hub cache AND run a tiny inference, so
  # setup fails HERE (not at first real use) if the --no-deps install is missing an
  # inference-path dep (utils3d / cv2 / scipy). Idempotent (from_pretrained reuses the
  # snapshot) and ungated (MoGe is MIT). Model id: MOGE_MODEL_ID.
  echo "Pre-caching + smoke-testing MoGe-2 weights (${MOGE_MODEL_ID})..."
  MOGE_MODEL_ID="${MOGE_MODEL_ID}" python - <<'PYEOF'
import os
import torch
from moge.model.v2 import MoGeModel
mid = os.environ["MOGE_MODEL_ID"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = MoGeModel.from_pretrained(mid).to(dev).eval()
with torch.no_grad():
    out = model.infer(torch.rand(3, 256, 256, device=dev))
assert {"points", "depth", "mask", "intrinsics"} <= set(out), f"MoGe infer output missing keys: {set(out)}"
print("MoGe-2 weights cached + infer OK:", mid)
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
  # NOTE: `cmd | grep -q` is unsafe under `set -o pipefail`. grep -q exits at the
  # first match and closes the pipe, so the producer dies of SIGPIPE (141) and
  # pipefail reports the pipeline as FAILED even though the match succeeded.
  # This guard therefore never fired, and ffmpeg was reinstalled on every run.
  # Capture first, match second.
  if [[ -x "${ffmpeg_bin}" ]]; then
    local ff_ver ff_enc
    ff_ver="$("${ffmpeg_bin}" -version 2>/dev/null | head -n 1 || true)"
    ff_enc="$("${ffmpeg_bin}" -hide_banner -encoders 2>/dev/null || true)"
    if [[ "${ff_ver}" == *"ffmpeg version ${FFMPEG_VERSION}"* && "${ff_enc}" == *libx264* ]]; then
      echo "ffmpeg ${FFMPEG_VERSION} (with libx264) already present at ${ffmpeg_bin} — skipping."
      return
    fi
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
      check_env_python_version
      return
    fi

    local existing_envs
    existing_envs="$(conda env list | awk '{print $1}' || true)"
    if ! grep -qx "${target_env}" <<<"${existing_envs}"; then
      # Creating an env is a multi-GB, multi-minute operation, so it is opt-in:
      # without --create-env a typo in --conda-env fails here rather than
      # silently building an environment the caller never asked for.
      if [[ $CREATE_ENV -eq 0 ]]; then
        echo "Conda environment '${target_env}' does not exist." >&2
        echo "Re-run with --create-env to create it, or pass an existing --conda-env NAME." >&2
        exit 1
      fi
      local create_py
      create_py="$(lockfile_python_minor)"
      if [[ -z "${create_py}" ]]; then
        echo "Could not read a Python version from ${REIFYSCENE_LOCK_FILE}." >&2
        echo "Fix the lockfile header, or use --conda-env with an existing env." >&2
        exit 1
      fi
      echo "Creating conda env '${target_env}' with python=${create_py} (from lockfile)..."
      if ! conda create -y -n "${target_env}" "python=${create_py}"; then
        echo "Failed to create conda environment '${target_env}'." >&2
        exit 1
      fi
    fi

    echo "Activating conda environment: ${target_env}"
    # conda's activate.d hooks are not written to be `set -u` safe: the
    # cuda-nvcc hook expands ${NVCC_PREPEND_FLAGS} unconditionally, so under
    # `set -euo pipefail` activation aborts with "unbound variable" — but only
    # once cuda-nvcc is installed. That made this script runnable exactly ONCE
    # per environment: the first run activates a bare env and installs the hook,
    # and every later run dies on activation. Relax nounset across activation
    # only, then restore it.
    set +u
    if ! conda activate "${target_env}"; then
      set -u
      echo "Failed to activate conda environment '${target_env}'." >&2
      exit 1
    fi
    set -u
    check_env_python_version
    return
  fi

  if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "Using already-active conda environment: ${CONDA_DEFAULT_ENV}"
    return
  fi

  echo "No active conda environment detected. Provide --conda-env NAME or use --use-venv." >&2
  exit 1
}

# Read the Python version the lockfile was captured under, from its header:
#   "# Platform: Linux/x86_64 | Python 3.11.15 | torch ..."
# lockfile_python_full  -> 3.11.15   lockfile_python_minor -> 3.11
# Install a package while honouring the pinned tree.
#
# TRANSFORMERS_SPEC is an open-ended lower bound (`transformers>=4.44`) and
# transformers is NOT in the lockfile, so these installs run *after* the locked
# tree and silently escape it: a fresh machine resolves whatever is newest that
# day. That is how a from-scratch env ended up with transformers 5.17.0 against
# the lockfile's safetensors==0.7.0 — a combination transformers rejects at
# import, breaking every test that touches it, while the older working env sat
# happily on 5.12.1. Passing the lockfile as a pip constraints file makes the
# resolver honour the existing pins and pick a compatible version instead.
pip_install_constrained() {
  if [[ $UNLOCKED -eq 0 && -f "${REIFYSCENE_LOCK_FILE}" ]]; then
    python -m pip install -c "${REIFYSCENE_LOCK_FILE}" "$@"
  else
    python -m pip install "$@"
  fi
}

lockfile_python_full() {
  [[ -f "${REIFYSCENE_LOCK_FILE}" ]] || return 0
  sed -n 's/.*| *Python \([0-9][0-9.]*\) *|.*/\1/p' "${REIFYSCENE_LOCK_FILE}" | head -n 1
}

lockfile_python_minor() {
  local full; full="$(lockfile_python_full)"
  [[ -n "${full}" ]] || return 0
  echo "${full}" | cut -d. -f1,2
}

# Compare the active interpreter against the lockfile.
#
# MAJOR.MINOR must match: wheels are tagged by minor version (cp311), so a
# 3.11-pinned tree simply will not install on 3.12. The PATCH level is not part
# of the ABI, so a difference there is reported and allowed — requiring an exact
# patch match would fail on a perfectly good env every time Python is updated.
check_env_python_version() {
  local want_minor want_full have_full have_minor
  want_minor="$(lockfile_python_minor)"
  want_full="$(lockfile_python_full)"
  [[ -n "${want_minor}" ]] || return 0
  have_full="$(python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null)" || return 0
  have_minor="$(echo "${have_full}" | cut -d. -f1,2)"

  if [[ "${have_minor}" != "${want_minor}" ]]; then
    echo "Python version mismatch: env has ${have_full}, lockfile pins ${want_full}." >&2
    echo "  The pinned wheels are built for Python ${want_minor} and will not install on ${have_minor}." >&2
    echo "  Use an env with Python ${want_minor}, or re-resolve with --unlocked." >&2
    exit 1
  fi
  if [[ "${have_full}" != "${want_full}" ]]; then
    echo "Note: env Python ${have_full}, lockfile captured under ${want_full} (same ${want_minor} series — fine)."
  fi
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
  # Bootstrap the build frontend, but do NOT let setuptools float to whatever is
  # newest today: this runs before the pinned tree, and an unbounded upgrade is
  # how a fresh env ended up on setuptools 84 while the pinned torch declares
  # `setuptools<82` (the long-lived env sat on 81 only because that was current
  # when it was built). Bounded by the same constraint torch states; raise it
  # when the pinned torch does. Same failure mode as the transformers/safetensors
  # escape — see pip_install_constrained().
  python -m pip install --upgrade pip wheel "setuptools${SETUPTOOLS_MAX_SPEC}"
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
  # Capture then match — see the pipefail/SIGPIPE note in ensure_ffmpeg().
  local encoders
  encoders="$("${ffmpeg_bin}" -hide_banner -encoders 2>/dev/null || true)"
  if [[ "${encoders}" != *libx264* ]]; then
    echo "[smoke-test] FAIL ffmpeg: libx264 encoder missing (need the GPL build for -crf 0 H.264)" >&2
    echo "  probed: ${ffmpeg_bin}" >&2
    exit 1
  fi
  local ver
  ver="$("${ffmpeg_bin}" -version 2>/dev/null | head -n 1)"
  echo "[smoke-test] OK ffmpeg: ${ver} path=${ffmpeg_bin} (libx264 present)"
}

run_post_install_smoke_tests() {
  echo
  echo "Running post-install smoke tests..."
  smoke_test_python_module "reifyscene" "reifyscene" "reifyscene"

  if [[ $INSTALL_FFMPEG -eq 1 ]]; then
    smoke_test_ffmpeg
  fi

  if [[ $INSTALL_DETECTRON2 -eq 1 ]]; then
    smoke_test_python_module "detectron2" "detectron2" "detectron2"
  fi
  if [[ $INSTALL_VGGT -eq 1 ]]; then
    smoke_test_python_module "VGGT" "vggt" "vggt"
  fi
  if [[ $INSTALL_VGGT_OMEGA -eq 1 ]]; then
    smoke_test_python_module "VGGT-Omega" "vggt_omega" "vggt-omega"
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
# editable reifyscene install below then finds every dependency already satisfied
# at the exact known-good version instead of re-resolving ranges.
if [[ $UNLOCKED -eq 0 && -f "${REIFYSCENE_LOCK_FILE}" ]]; then
  echo "Installing pinned dependency tree from ${REIFYSCENE_LOCK_FILE} (use --unlocked to re-resolve)..."
  python -m pip install -r "${REIFYSCENE_LOCK_FILE}" \
    --extra-index-url "${TORCH_INDEX_URL}" \
    --extra-index-url "https://pypi.nvidia.com"
elif [[ $UNLOCKED -eq 0 ]]; then
  echo "No lockfile at ${REIFYSCENE_LOCK_FILE} — falling back to fresh resolution."
fi

EXTRAS=()
[[ $INSTALL_VISION -eq 1 ]] && EXTRAS+=(vision)
[[ $INSTALL_GEOMETRY -eq 1 ]] && EXTRAS+=(geometry)
[[ $INSTALL_CALIBRATION -eq 1 ]] && EXTRAS+=(calibration)
[[ $INSTALL_POSE_REFINE -eq 1 ]] && EXTRAS+=(pose_refine)
[[ $INSTALL_VGGT_LONG -eq 1 ]] && EXTRAS+=(vggt_long)
[[ $INSTALL_DEV -eq 1 ]] && EXTRAS+=(dev)

# Install reifyscene and all Python dependencies from pyproject.toml only.
if [[ ${#EXTRAS[@]} -gt 0 ]]; then
  EXTRA_CSV=$(IFS=,; echo "${EXTRAS[*]}")
  python -m pip install -e "${ROOT_DIR}[${EXTRA_CSV}]"
else
  python -m pip install -e "${ROOT_DIR}"
fi

if [[ $INSTALL_DETECTRON2 -eq 1 ]]; then
  # OFF BY DEFAULT and not part of the standard install. detectron2 exists here
  # for DensePose, intended as the dense top rung whenever Sapiens is ruled out:
  # Sapiens is CC-BY-NC-4.0 (non-commercial), its weights are gated behind manual
  # approval, and it is a large ViT. DensePose is Apache-2.0, ungated and lighter,
  # so it is the PERMISSIVELY-LICENSED dense rung. Nothing consumes it yet:
  # densepose_wrapper.py reports `unavailable` unconditionally and the dense
  # ladder has no DensePose rung, so wiring one is follow-up work.
  #
  # Known caveat when you DO enable it: detectron2 caps iopath<0.1.10 while SAM-2
  # requires >=0.1.10, so installing it downgrades iopath and pip check reports the
  # conflict. SAM-2 uses iopath only for `g_pathmgr` and runs fine on 0.1.9, but
  # the declared constraint is genuinely unsatisfiable — resolve it properly when
  # the DensePose rung lands.
  #
  # Installed NON-editable, unlike the third_party checkouts. Those are editable
  # because the installer patches them and we want the patched tree to be the
  # imported one; detectron2 is used unmodified. It was previously
  # `pip install -e` against a mktemp -d clone, which left the module permanently
  # resolving into /tmp — so the env broke as soon as that directory was reaped,
  # and the temp tree leaked because nothing removed it.
  echo "Installing detectron2 from main @ ${DETECTRON2_COMMIT:0:12}..."
  # Installed straight from the pinned tag rather than a local clone: a
  # `mktemp -d` clone left the module resolving into a temporary folder (editable)
  # or recorded a dead file:// direct_url (non-editable, which then leaked into the
  # lockfile and made it unportable), and the temp tree leaked either way.
  # Like gsplat, detectron2 is excluded from the lock — it is owned by this
  # script, pinned to a tag, and needs --no-build-isolation.
  python -m pip install --no-build-isolation "git+${DETECTRON2_REPO}@${DETECTRON2_COMMIT}"
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
  pip_install_constrained "${TRANSFORMERS_SPEC}"
  download_dinov2_weights_if_needed
fi

if [[ $INSTALL_SIGLIP2 -eq 1 ]]; then
  echo "Installing HuggingFace transformers (${TRANSFORMERS_SPEC}) for the SigLIP2 semantics rung..."
  pip_install_constrained "${TRANSFORMERS_SPEC}"
  download_siglip2_weights_if_needed
fi

if [[ $INSTALL_SAPIENS -eq 1 ]]; then
  download_sapiens_checkpoints_if_needed
fi

if [[ $INSTALL_MOGE -eq 1 ]]; then
  clone_or_update_checkout "${MOGE_REPO}" "${MOGE_DIR}" "${MOGE_COMMIT}"
  echo "Installing MoGe-2 from ${MOGE_DIR} (--no-deps: keeps its unpinned torchvision + GUI-only gradio off the locked tree)..."
  # MoGe's INFERENCE path (verified by tracing `from moge.model.v2 import MoGeModel`
  # + a smoke infer) needs utils3d (its own geometry helpers), opencv-python (cv2),
  # and scipy; matplotlib is NOT on the infer path and gradio/pipeline are
  # GUI/training-only. cv2/scipy happen to ship via the default-on vision/geometry
  # extras, but --with-moge should not silently depend on other flags — so add MoGe's
  # real infer deps explicitly here. utils3d is --no-deps so it can't drag torch
  # (mirrors the gsplat guard); opencv/scipy are already-present leaves (no churn).
  python -m pip install --no-deps -e "${MOGE_DIR}"
  python -m pip install --no-deps "${MOGE_UTILS3D_SPEC}"
  python -m pip install "opencv-python>=4.10" "scipy>=1.13"
  download_moge_weights_if_needed
fi

if [[ $INSTALL_CALIBRATION -eq 1 ]]; then
  clone_or_update_checkout "${HLOC_REPO}" "${HLOC_DIR}" "${HLOC_TAG}"
  if [[ -f "${HLOC_DIR}/.gitmodules" ]]; then
    git -C "${HLOC_DIR}" submodule update --init --recursive
  fi
  echo "Installing hloc from ${HLOC_DIR} (tag ${HLOC_TAG})..."
  python -m pip install -e "${HLOC_DIR}"

  # Patch hloc for pycolmap >= 3.10 API drift; see the patch header for the three call sites.
  echo "Patching hloc for pycolmap >= 3.10 API drift..."
  apply_patch_series "${HLOC_DIR}" "${ROOT_DIR}/third_party_patches/hloc" "hloc"
fi

if [[ $INSTALL_VLA3D -eq 1 ]]; then
  # Relations for the scene graph. Their code is imported from the checkout, not pip-installed;
  # shapely and numba are what it needs (the `relations` extra in pyproject.toml).
  clone_or_update_checkout "${VLA3D_REPO}" "${VLA3D_DIR}" "${VLA3D_COMMIT}"
  python -m pip install "shapely>=2.0" "numba>=0.60"
  echo "Applying VLA-3D patches..."
  apply_patch_series "${VLA3D_DIR}" "${ROOT_DIR}/third_party_patches/vla-3d" "VLA-3D"
fi

if [[ $INSTALL_VGGT_LONG -eq 1 ]]; then
  ensure_modern_conda_compilers
  clone_or_update_checkout "${VGGT_LONG_REPO}" "${VGGT_LONG_DIR}" "${VGGT_LONG_COMMIT}"
  install_vggt_long_requirements
  ensure_faiss
  ensure_cuda_toolkit_for_torch_extensions

  # Two VGGT-Long patches, both in third_party_patches/vggt-long/: the loop-closure DINOv2
  # backbone's CWD-relative torch.hub path anchored to __file__, and the weight-download script
  # moved onto the current Hugging Face CLI. See each patch header for why.
  echo "Applying VGGT-Long patches..."
  apply_patch_series "${VGGT_LONG_DIR}" "${ROOT_DIR}/third_party_patches/vggt-long" "VGGT-Long"

  # Provide a compatibility shim if the environment has `hf` but not `huggingface-cli`.
  if ! command -v huggingface-cli >/dev/null 2>&1 && command -v hf >/dev/null 2>&1; then
    mkdir -p "${CONDA_PREFIX:-${ROOT_DIR}/.reifyscene_cache}/bin"
    cat > "${CONDA_PREFIX:-${ROOT_DIR}/.reifyscene_cache}/bin/huggingface-cli" <<'EOF'
#!/usr/bin/env bash
exec hf "$@"
EOF
    chmod +x "${CONDA_PREFIX:-${ROOT_DIR}/.reifyscene_cache}/bin/huggingface-cli"
    export PATH="${CONDA_PREFIX:-${ROOT_DIR}/.reifyscene_cache}/bin:${PATH}"
  fi

  download_vggt_long_weights_if_needed

  echo "Installing VGGT-Long from ${VGGT_LONG_DIR}..."
  MAX_JOBS="${MAX_JOBS:-4}" python -m pip install --no-build-isolation -e "${VGGT_LONG_DIR}"
fi

if [[ $INSTALL_VGGT_OMEGA -eq 1 ]]; then
  clone_or_update_checkout "${VGGT_OMEGA_REPO}" "${VGGT_OMEGA_DIR}" "${VGGT_OMEGA_COMMIT}"
  echo "Installing VGGT-Omega from ${VGGT_OMEGA_DIR} (no dependencies: the lock pins them)..."
  python -m pip install --no-deps -e "${VGGT_OMEGA_DIR}"
  echo "VGGT-Omega's weights (facebook/VGGT-Omega) are gated: request access on Hugging Face and"
  echo "run 'hf auth login'; the checkpoint downloads the first time the backend runs."
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
- reifyscene in editable mode from:
    ${ROOT_DIR}
- Python dependencies from pyproject.toml
$( [[ $INSTALL_DETECTRON2 -eq 1 ]] && echo "- detectron2 from main @ ${DETECTRON2_COMMIT:0:12}" )
$( [[ $INSTALL_VGGT -eq 1 ]] && echo "- VGGT from commit ${VGGT_COMMIT} at ${VGGT_DIR}" )
$( [[ $INSTALL_COTRACKER -eq 1 ]] && echo "- CoTracker from commit ${COTRACKER_COMMIT} at ${COTRACKER_DIR}" )
$( [[ $INSTALL_SAM2 -eq 1 ]] && echo "- SAM 2 from commit ${SAM2_COMMIT} at ${SAM2_DIR} (+ sam2.1_hiera_large checkpoint)" )
$( [[ $INSTALL_DINOV2 -eq 1 ]] && echo "- transformers (${TRANSFORMERS_SPEC}) + DINOv2-L weights ${DINOV2_MODEL_ID} cached (FP3 reid rung)" )
$( [[ $INSTALL_SIGLIP2 -eq 1 ]] && echo "- transformers (${TRANSFORMERS_SPEC}) + SigLIP2 weights ${SIGLIP2_MODEL_ID} cached (FP4 semantics rung)" )
$( [[ $INSTALL_SAPIENS -eq 1 ]] && echo "- Sapiens ${SAPIENS_SIZE} dense checkpoints cached (FP5 humans rung; CC-BY-NC-4.0)" )
$( [[ $INSTALL_MOGE -eq 1 ]] && echo "- MoGe-2 from commit ${MOGE_COMMIT} at ${MOGE_DIR} (--no-deps + utils3d) + weights ${MOGE_MODEL_ID} cached (FP6 geometry depth backstop; MIT)" )
$( [[ $INSTALL_CALIBRATION -eq 1 ]] && echo "- calibration extras from pyproject.toml (pycolmap ${PYCOLMAP_VERSION}, matching COLMAP ${COLMAP_UPSTREAM_TAG})" )
$( [[ $INSTALL_CALIBRATION -eq 1 ]] && echo "- hloc from tag ${HLOC_TAG} at ${HLOC_DIR}" )
$( [[ $INSTALL_VGGT_LONG -eq 1 ]] && echo "- VGGT-Long from commit ${VGGT_LONG_COMMIT} at ${VGGT_LONG_DIR}" )
$( [[ $INSTALL_VGGT_LONG -eq 1 ]] && echo "- VGGT-Long loop closure enabled: faiss-cpu ${FAISS_VERSION} + DINOv2 path patch (CWD-relative -> __file__)" )
$( [[ $INSTALL_GSPLAT -eq 1 ]] && echo "- gsplat built from source for sm_120 at commit ${GSPLAT_REF}" )
$( [[ $INSTALL_FFMPEG -eq 1 ]] && echo "- ffmpeg ${FFMPEG_VERSION} (conda-forge GPL build with libx264)" )

Notes:
- third_party/ should be gitignored so external source checkouts are never tracked.
- .reifyscene_cache/ should be gitignored too.
- Calibration mode is pinned to hloc ${HLOC_TAG} and pycolmap ${PYCOLMAP_VERSION}
  (aligned with upstream COLMAP ${COLMAP_UPSTREAM_TAG}).
- Post-install smoke tests verify the third-party repos and pinned Python packages
  this script installs directly into the active environment.
- For Intel RealSense SDK / pyrealsense2 on Ubuntu, see:
    scripts/setup_realsense_sdk.sh
- For a first RGB-only camera test without RealSense SDK, you can use:
    python scripts/capture_uvc_video.py --device 0 --output out.mp4
EOF
