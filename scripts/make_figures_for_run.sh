#!/bin/bash
# =============================================================================
# make_figures_for_run.sh
#
# Compute + save every analysis figure's .npz AND .png INTO an experiment run's
# output folder (experiments/<EXP>/pred/<out_name>/), one artifact PER DATASET
# plus one COMBINED artifact across all datasets in the run.
#
# It reuses the existing, UNMODIFIED figure_making_script/*/compute_*.py and
# plot_*.py by passing explicit --out / --data paths (so nothing there changes).
# The figures are computed with the *same* hyper-parameters as the run: this
# script synthesizes a .conf from the per-combo FIG_* values the caller exports.
#
# Invoked (gated by MAKE_FIGURES=1) at the end of the causal_lookahead branch of
# experiments/<EXP>/run.sh, or standalone:
#
#   FIG_BENCHMARK=longbench FIG_BUDGET=128 \
#   FIG_RUN_DIR=/abs/experiments/LongBench/pred/<out_name> \
#   FIG_DATASETS="qasper narrativeqa multifieldqa_en" \
#   FIG_NUM_PARTICLES=1 FIG_LOOKAHEAD_LEN=8 FIG_LAMBDA_LOOKAHEAD=1.0 ... \
#   bash scripts/make_figures_for_run.sh
#
# COST WARNING: each (figure x dataset) is a separate process that reloads the
# model. With many datasets/figures this is heavy — keep FIG_SAMPLES small and
# use FIG_LIST / FIG_DATASETS to scope it. This is why it is opt-in.
# =============================================================================
set -uo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SELF/.." && pwd)"

# --- Load central path/env configuration (.env) ---
set -a
[[ -f "$REPO_ROOT/.env" ]] && source "$REPO_ROOT/.env"
set +a

# Reduce CUDA fragmentation across long/variable-length samples.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Cap analysis prompt length (middle-truncation) so very long samples don't OOM. 0 = off.
export FIG_MAX_ANALYSIS_TOKENS="${FIG_MAX_ANALYSIS_TOKENS:-24000}"

BM="${FIG_BENCHMARK:-longbench}"
BUDGET="${FIG_BUDGET:-128}"
RUN_DIR="${FIG_RUN_DIR:?FIG_RUN_DIR (the run output folder) is required}"
SAMPLES="${FIG_SAMPLES:-2}"
FIGS="${FIG_LIST:-1 2 3 4 5 6 7 8 9 10}"
GPU="${FIG_GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
read -r -a DATASETS <<< "${FIG_DATASETS:?FIG_DATASETS (space-separated) is required}"

case "$BM" in
  ruler) FIG_DIR="$REPO_ROOT/experiments/RULER/figure_making_script"; BM="ruler" ;;
  *)     FIG_DIR="$REPO_ROOT/experiments/LongBench/figure_making_script"; BM="longbench" ;;
esac
mkdir -p "$RUN_DIR"

# --- Synthesize the exact run config so figures use the run's hyper-parameters ---
# FIG_CONFIG overrides this if the caller wants a fixed .conf instead.
CFG="${FIG_CONFIG:-}"
if [[ -z "$CFG" ]]; then
  CFG="$RUN_DIR/figure_config.conf"
  cat > "$CFG" <<EOF
# Auto-generated from the run's per-combo hyper-parameters (do not edit).
MODEL_PATH            = ${MODEL_PATH:-mistralai/Mistral-7B-Instruct-v0.2}
CHUNK_SIZE            = ${FIG_CHUNK:-4}
LAMBDA_RED            = ${FIG_LAMBDA_RED:-0.0}
NUM_PARTICLES         = ${FIG_NUM_PARTICLES:-4}
LOOKAHEAD_LEN         = ${FIG_LOOKAHEAD_LEN:-8}
LOOKAHEAD_TEMPERATURE = ${FIG_LOOKAHEAD_TEMPERATURE:-1.0}
BETA                  = ${FIG_BETA:-1.0}
GAMMA                 = ${FIG_GAMMA:-0.0}
LAMBDA_D              = ${FIG_LAMBDA_D:-0.0}
NORMALIZE_TYPICALITY  = ${FIG_NORMALIZE_TYPICALITY:-0}
ROBUST_ALPHA          = ${FIG_ROBUST_ALPHA:-0.0}
ROBUST_TOPK_BETA      = ${FIG_ROBUST_TOPK_BETA:-1.0}
LAMBDA_LOOKAHEAD      = ${FIG_LAMBDA_LOOKAHEAD:-1.0}
LAMBDA_K              = ${FIG_LAMBDA_K:-0.0}
LAMBDA_V              = ${FIG_LAMBDA_V:-0.0}
BUDGET_TAU            = ${FIG_BUDGET_TAU:-1.0}
EOF
fi

echo "=============================================================="
echo " make_figures_for_run: BM=$BM BUDGET=$BUDGET SAMPLES=$SAMPLES"
echo "   RUN_DIR=$RUN_DIR"
echo "   DATASETS=${DATASETS[*]}"
echo "   CONFIG=$CFG   FIGS=$FIGS"
echo "=============================================================="

want() { [[ " $FIGS " == *" $1 "* ]]; }

# id -> "subdir|compute|plot|data_base|fig_base|extra-compute-args"
fig_meta() {
  case "$1" in
    1) echo "fig1_future_query_alignment|compute_alignment|plot_alignment|alignment_data|fig1_future_query_alignment|--budget $BUDGET --m-list 1 2 4 8 --decode-steps 32" ;;
    2) echo "fig2_score_comparison|compute_scores|plot_scores|score_data|fig2_score_comparison|--budget $BUDGET" ;;
    3) echo "fig3_cache_distribution|compute_distribution|plot_distribution|distribution_data|fig3_cache_distribution|--budget $BUDGET --decode-steps 16" ;;
    4) echo "fig4_entropy_trajectory|compute_entropy|plot_entropy|entropy_data|fig4_entropy_trajectory|--budget $BUDGET --num-particles 8 --decode-steps 32" ;;
    5) echo "fig5_confidence_calibration|compute_calibration|plot_calibration|calibration_data|fig5_confidence_calibration|--budget $BUDGET --num-particles 8 --decode-steps 32" ;;
    6) echo "fig6_mismatch_vs_gain|compute_mismatch|plot_mismatch|mismatch_data|fig6_mismatch_vs_gain|--budget $BUDGET --decode-steps 24" ;;
    7) echo "fig7_retained_token_composition|compute_composition|plot_composition|composition_data|fig7_retained_token_composition|--budget $BUDGET" ;;
    8) echo "fig8_layer_head_analysis|compute_layer_head|plot_layer_head|layer_head_data|fig8_layer_head_analysis|--budget $BUDGET --decode-steps 24" ;;
    9) echo "fig9_budget_equivalence|compute_budget|plot_budget|budget_data|fig9_budget_equivalence|--budgets 64 128 256 512 1024 --score-budget $BUDGET --decode-steps 24" ;;
    10) echo "fig10_particle_divergence|compute_divergence|plot_divergence|divergence_data|fig10_particle_divergence|--budget $BUDGET --num-particles 8 --decode-steps 32" ;;
  esac
}

# figure_id  dataset_arg  label
run_one() {
  local id="$1" dsarg="$2" label="$3"
  local meta subdir compute plot data_base fig_base extra
  meta="$(fig_meta "$id")"; [[ -z "$meta" ]] && { echo "!!! unknown figure $id"; return; }
  IFS='|' read -r subdir compute plot data_base fig_base extra <<< "$meta"
  local npz="$RUN_DIR/${data_base}__${label}.npz"
  local png="$RUN_DIR/${fig_base}__${label}.png"
  echo ">>> figure $id [$label]  ($dsarg)"
  # shellcheck disable=SC2086
  if CUDA_VISIBLE_DEVICES="$GPU" python "$FIG_DIR/$subdir/${compute}.py" \
        --benchmark "$BM" --config "$CFG" --dataset "$dsarg" \
        --samples-per-dataset "$SAMPLES" $extra --out "$npz"; then
    CUDA_VISIBLE_DEVICES="$GPU" python "$FIG_DIR/$subdir/${plot}.py" --data "$npz" --out "$png" \
      || echo "!!! figure $id [$label] plot failed (continuing)"
  else
    echo "!!! figure $id [$label] compute failed (continuing)"
  fi
}

for id in $FIGS; do
  want "$id" || continue
  for ds in "${DATASETS[@]}"; do
    run_one "$id" "$ds" "$ds"
  done
  if [ "${#DATASETS[@]}" -gt 1 ]; then
    joined="$(IFS=,; echo "${DATASETS[*]}")"
    run_one "$id" "$joined" "combined"
  fi
done

echo "make_figures_for_run: done -> $RUN_DIR"
exit 0
