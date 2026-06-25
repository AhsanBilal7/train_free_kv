"""
#!/bin/bash
#SBATCH -p sooner_gpu_test_h100
#SBATCH --gres=gpu:1
#SBATCH --job-name=andpro_longbench_h100
#SBATCH --output=/home/ahsan/train_kv/logs/out_andpro_h100_%j.log
#SBATCH --error=/home/ahsan/train_kv/logs/err_andpro_h100_%j.log
#SBATCH --time=2-00:00:00
"""

set -euo pipefail

source /scratch/ahsan/venvs/kv44/bin/activate

mkdir -p /home/ahsan/LLM-AnDPro/logs

export CUDA_VISIBLE_DEVICES=1
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
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HOME"

# --- Show which GPU(s) this job actually got ---
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
export GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
echo "Using GPU_ID=$GPU_ID"

# REPO_DIR="${REPO_DIR:-/scratch/$USER/LLM-AnDPro}"
# if [[ ! -d "$REPO_DIR" ]]; then
#     git clone git@github.com:MIRALab-USTC/LLM-AnDPro.git "$REPO_DIR"
# fi
# cd "$REPO_DIR"

# # Build the CUDA extension (sm_90 for H100, picked up automatically since
# # csrc/build.py now adds the sm_90 gencode whenever CUDA >= 11.8) and
# # install the package. Only needed once per venv -- skip if already installed.
# if ! python -c "import tiny_api_cuda" >/dev/null 2>&1; then
#     make i
# fi

# --- Path Configuration (defaults pull straight from the Hugging Face Hub) ---
export MODEL_PATH="${MODEL_PATH:-mistralai/Mistral-7B-Instruct-v0.2}"
export DATASET_PATH="${DATASET_PATH:-THUDM/LongBench}"

cd experiments/LongBench
bash run.sh
