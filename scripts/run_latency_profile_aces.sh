#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=latency-profile
#SBATCH --output=logs/out_latency_profile_%j.log
#SBATCH --error=logs/err_latency_profile_%j.log

# One-off job for scripts/profile_latency_breakdown.py -- NOT part of the
# lorekv_orchestrator queue. See that script's docstring for what it measures.
#
#   sbatch scripts/run_latency_profile_aces.sh

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
cd "$REPO_ROOT"

set -a
[[ -f "$REPO_ROOT/.env" ]] && source "$REPO_ROOT/.env"
set +a

mkdir -p logs analysis
export TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"
export HF_HUB_OFFLINE=1
mkdir -p "$TMPDIR"

module load WebProxy GCC/11.3.0 CUDA/12.1.1
source "${VENV_PATH:?VENV_PATH not set — check .env}/bin/activate"

nvidia-smi -L || true

cd "$REPO_ROOT/experiments/LongBench"
python ../../scripts/profile_latency_breakdown.py \
    --model_path mistralai/Mistral-7B-Instruct-v0.2 \
    --methods andpro critical lorekv \
    --budgets 128 256 512 1024 \
    --num_samples "${NUM_SAMPLES:-10}" \
    --lorekv_m "${LOREKV_M:-4}" --lorekv_t "${LOREKV_T:-4}" \
    --datasets longbench ruler niah \
    --out "$REPO_ROOT/analysis/latency_breakdown.json"
