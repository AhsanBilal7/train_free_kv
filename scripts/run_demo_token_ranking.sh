#!/bin/bash
# =============================================================================
# Token-ranking demo launcher (ACES / H100)
#
# Runs scripts/demo_token_ranking.py on LongBench and RULER and writes every
# artifact under demos/<run>/ for the comparison figures (CriticalKV vs AnDPro
# vs LORE-KV token ranking).
#
# Submit:
#   sbatch scripts/run_demo_token_ranking.sh
#   sbatch --export=ALL,SAMPLES=4,BUDGETS="128 256 512" scripts/run_demo_token_ranking.sh
#
# Or inside an existing allocation / an ssh session on the compute node:
#   cd ~/train_free_kv && bash scripts/run_demo_token_ranking.sh
#
# Overridables: CONFIG_FILE MODEL_PATH SAMPLES BUDGETS DECODE_STEPS MAX_LENGTH
#               LB_DATASETS RULER_DATASETS BENCHMARKS OUT_DIR
# =============================================================================
#SBATCH --job-name=lorekv-demo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=/home/u.ab353540/train_free_kv/logs/out_%x_%j.log
#SBATCH --error=/home/u.ab353540/train_free_kv/logs/err_%x_%j.log
#SBATCH --account=158535403388

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    module purge
    module load WebProxy
    module load GCC/11.3.0
    module load CUDA/12.1.1
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}}"
cd "$REPO_ROOT"

if [[ -z "${SCRATCH:-}" && -d "/scratch/user/$USER" ]]; then
    export SCRATCH="/scratch/user/$USER"
fi
# .env exports MODEL_PATH unconditionally (mistral by default), so a config's own
# MODEL_PATH would be silently overridden -- exactly the bug scripts/run_h100.sh
# documents. Record whether the CALLER set it, before .env is sourced.
_USER_MODEL="${MODEL_PATH:-}"
[[ -f "$REPO_ROOT/.env" ]] && source "$REPO_ROOT/.env"
source "${VENV_PATH:?VENV_PATH not set — check .env}/bin/activate"

CONFIG_FILE="${CONFIG_FILE:-config/lorekv-ent-unif-topk.conf}"
[[ "$CONFIG_FILE" != */* && "$CONFIG_FILE" != *.conf ]] && CONFIG_FILE="config/${CONFIG_FILE}.conf"
[[ "$CONFIG_FILE" != /* ]] && CONFIG_FILE="$REPO_ROOT/$CONFIG_FILE"

SAMPLES="${SAMPLES:-3}"
BUDGETS="${BUDGETS:-128 512}"
DECODE_STEPS="${DECODE_STEPS:-24}"
MAX_LENGTH="${MAX_LENGTH:-16000}"
LB_DATASETS="${LB_DATASETS:-single}"
RULER_DATASETS="${RULER_DATASETS:-4096,8192}"
BENCHMARKS="${BENCHMARKS:-longbench ruler}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/demos}"

echo "=============================================================="
echo "LORE-KV token-ranking demo on $(hostname)"
echo "  CONFIG      = $CONFIG_FILE"
echo "  MODEL_PATH  = ${_USER_MODEL:-<from config>}"
echo "  BENCHMARKS  = $BENCHMARKS"
echo "  SAMPLES     = $SAMPLES   BUDGETS = $BUDGETS"
echo "  OUT_DIR     = $OUT_DIR"
echo "=============================================================="

# Only an explicit caller-supplied MODEL_PATH overrides the config's own value.
model_args=()
[[ -n "$_USER_MODEL" ]] && model_args=(--model "$_USER_MODEL")

for bench in $BENCHMARKS; do
    if [[ "$bench" == "ruler" ]]; then
        ds="$RULER_DATASETS"
    else
        ds="$LB_DATASETS"
    fi
    echo "===== $bench / $ds ====="
    python "$REPO_ROOT/scripts/demo_token_ranking.py" \
        --config "$CONFIG_FILE" "${model_args[@]}" \
        --benchmark "$bench" --dataset "$ds" \
        --samples-per-dataset "$SAMPLES" \
        --budgets $BUDGETS \
        --decode-steps "$DECODE_STEPS" \
        --max-length "$MAX_LENGTH" \
        --out-dir "$OUT_DIR"
done

echo "demo artifacts under $OUT_DIR"
