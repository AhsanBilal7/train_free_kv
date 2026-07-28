# !/bin/bash
# SBATCH -p gpu
# SBATCH --gres=gpu:A100:1
# SBATCH --constraint=public_a100
# SBATCH --job-name=andpro_longbench_a100
# SBATCH --output=logs/out_andpro_a100_%j.log
# SBATCH --error=logs/err_andpro_a100_%j.log
# SBATCH --time=2-00:00:00

set -euo pipefail

# --- Load central path/env configuration (.env) ---
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
[[ -f "$HERE/../.env" ]] && source "$HERE/../.env"
set +a

mkdir -p logs

# Redirect every cache/tmp dir to scratch -- nothing should be written under $HOME.
export TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/$USER/pip_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/$USER/.cache}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/scratch/$USER/torch_extensions}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-/scratch/$USER/.nv/ComputeCache}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH"

export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME}"
mkdir -p "$HF_HOME"

# --- Show which GPU(s) this job actually got ---
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
export GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
echo "Using GPU_ID=$GPU_ID"

module load Python/3.10.8-GCCcore-12.2.0
# A100 uses the crisp_310 venv — uncomment that VENV_PATH line in .env.
source "${VENV_PATH:?VENV_PATH not set — check .env}/bin/activate"

REPO_DIR="${REPO_DIR:-/scratch/$USER/LLM-AnDPro}"
if [[ ! -d "$REPO_DIR" ]]; then
    git clone git@github.com:MIRALab-USTC/LLM-AnDPro.git "$REPO_DIR"
fi
cd "$REPO_DIR"

# Build the CUDA extension (sm_80 for A100) and install the package.
# Only needed once per venv -- skip if already installed.
if ! python -c "import tiny_api_cuda" >/dev/null 2>&1; then
    make i
fi

# --- Path Configuration (defaults pull straight from the Hugging Face Hub) ---
export MODEL_PATH="${MODEL_PATH:-mistralai/Mistral-7B-Instruct-v0.2}"
export DATASET_PATH="${DATASET_PATH:-THUDM/LongBench}"

cd experiments/LongBench
bash run.sh
