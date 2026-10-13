#!/bin/bash
#SBATCH --job-name=lorekv-demo
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Token-ranking analysis (paper App. F.3-F.5; Figures 9-11): CriticalKV vs AnDPro
# vs LORE-KV scored on the same captured prefill state, compared against the
# future-query oracle. Writes demos/<run>/; draw the figures with
#   python scripts/plot_token_ranking_demo.py
#
#   bash scripts/run_demo_token_ranking.sh
#   SAMPLES=4 BUDGETS="128 256 512" sbatch scripts/run_demo_token_ranking.sh
#
# Overridable: CONFIG_FILE MODEL_PATH SAMPLES BUDGETS DECODE_STEPS MAX_LENGTH
#              LB_DATASETS RULER_DATASETS BENCHMARKS OUT_DIR
set -euo pipefail
_USER_MODEL="${MODEL_PATH:-}"          # only an explicit MODEL_PATH overrides the config
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
cd "$REPO_ROOT"

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
