#!/bin/bash
#SBATCH --job-name=lorekv-figures
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Mechanism-analysis figures (paper App. F; Figures 5, 6, 9, 11-13).
#
# Every figure under experiments/<BENCHMARK>/figure_making_script/figN_*/ is a
# compute_*.py (runs the model, saves a .npz next to itself) and a plot_*.py
# (reads the .npz, writes the .png). This script runs both for the requested
# figures. The paper uses LORE-KV /E on Mistral-7B at B = 128, on LongBench
# single-document QA and RULER-4K:
#
#   CONFIG_FILE=config/lorekv-ent-unif-topk.conf bash scripts/run_figures.sh
#   BENCHMARK=ruler DATASET=4096 bash scripts/run_figures.sh
#   FIGURES="6 8" SAMPLES=20 bash scripts/run_figures.sh       # subset, fewer samples
#   PLOT_ONLY=1 bash scripts/run_figures.sh                    # redraw from saved .npz
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

# Per-sample checkpoints of the compute stage (resumable after a crash).
export FIG_CKPT_DIR="${FIG_CKPT_DIR:-$REPO_ROOT/logs/fig_ckpt}"
# Middle-truncate analysis prompts beyond this many tokens so the prefill plus M
# cache clones fit on one GPU (0 = no cap).
export FIG_MAX_ANALYSIS_TOKENS="${FIG_MAX_ANALYSIS_TOKENS:-24000}"
mkdir -p "$FIG_CKPT_DIR"

# =============================================================================
# Figure-run configuration (override any of these via env vars / CLI)
# =============================================================================
export BENCHMARK="${BENCHMARK:-longbench}"          # longbench | ruler
export CONFIG_FILE="${CONFIG_FILE:-$REPO_ROOT/config/lorekv-ent-unif-topk.conf}"
export BUDGET="${BUDGET:-128}"                      # tightest budget is most diagnostic
export SAMPLES="${SAMPLES:--1}"                      # samples per dataset/sub-task (-1 = all samples)
export FIGURES="${FIGURES:-1 2 3 4 5 6 7 8 9 10}"   # which figures to (re)build
export PLOT_ONLY="${PLOT_ONLY:-0}"                  # 1 = skip compute, only re-draw

# Default dataset differs by benchmark: LongBench single-doc QA vs a RULER length.
if [ "$BENCHMARK" = "ruler" ]; then
    DATASET="${DATASET:-4096}"
    FIG_DIR="$REPO_ROOT/experiments/RULER/figure_making_script"
else
    DATASET="${DATASET:-single}"                    # 'single' | 'full' | comma-list of tasks
    FIG_DIR="$REPO_ROOT/experiments/LongBench/figure_making_script"
fi

echo "=============================================================="
echo " BENCHMARK=$BENCHMARK  DATASET=$DATASET  BUDGET=$BUDGET  SAMPLES=$SAMPLES"
echo " CONFIG_FILE=$CONFIG_FILE"
echo " FIG_CKPT_DIR=$FIG_CKPT_DIR"
echo " FIGURES=$FIGURES   PLOT_ONLY=$PLOT_ONLY"
echo "=============================================================="

# Flags shared by every compute_*.py (RULER wrappers accept & pass these through).
COMMON=(--benchmark "$BENCHMARK" --config "$CONFIG_FILE" --dataset "$DATASET" \
        --samples-per-dataset "$SAMPLES")

want() { [[ " $FIGURES " == *" $1 "* ]]; }

# =============================================================================
# Figure 1 — future-query alignment (multi-trajectory vs prompt-window/single/…)
# =============================================================================
if want 1; then
  echo "### Figure 1: future-query alignment"
  D="$FIG_DIR/fig1_future_query_alignment"
  [ "$PLOT_ONLY" = "1" ] || python "$D/compute_alignment.py" "${COMMON[@]}" \
        --budget "$BUDGET" --m-list 1 2 4 8 --decode-steps 32
  python "$D/plot_alignment.py"
fi

# =============================================================================
# Figure 2 — raw score comparison (attention vs single-lookahead vs LORE-KV multi)
# =============================================================================
if want 2; then
  echo "### Figure 2: raw score comparison"
  D="$FIG_DIR/fig2_score_comparison"
  [ "$PLOT_ONLY" = "1" ] || python "$D/compute_scores.py" "${COMMON[@]}" --budget "$BUDGET"
  python "$D/plot_scores.py"
fi

# =============================================================================
# Figure 3 — full-cache vs evicted-cache output distribution
# =============================================================================
if want 3; then
  echo "### Figure 3: full vs evicted cache distribution"
  D="$FIG_DIR/fig3_cache_distribution"
  [ "$PLOT_ONLY" = "1" ] || python "$D/compute_distribution.py" "${COMMON[@]}" \
        --budget "$BUDGET" --decode-steps 16
  python "$D/plot_distribution.py"
fi

# =============================================================================
# Figure 4 — why test-time compute (entropy vs single-trajectory alignment)
# =============================================================================
if want 4; then
  echo "### Figure 4: entropy / trajectory diagnostic"
  D="$FIG_DIR/fig4_entropy_trajectory"
  [ "$PLOT_ONLY" = "1" ] || python "$D/compute_entropy.py" "${COMMON[@]}" \
        --budget "$BUDGET" --num-particles 8 --decode-steps 32
  python "$D/plot_entropy.py"
fi

# =============================================================================
# Figure 5 — confidence-weight calibration
# =============================================================================
if want 5; then
  echo "### Figure 5: confidence-weight calibration"
  D="$FIG_DIR/fig5_confidence_calibration"
  [ "$PLOT_ONLY" = "1" ] || python "$D/compute_calibration.py" "${COMMON[@]}" \
        --budget "$BUDGET" --num-particles 8 --decode-steps 32
  python "$D/plot_calibration.py"
fi

# =============================================================================
# Figure 6 — query-mismatch vs improvement (per-example)
# =============================================================================
if want 6; then
  echo "### Figure 6: mismatch vs gain"
  D="$FIG_DIR/fig6_mismatch_vs_gain"
  # more samples help the scatter/regression (but honor SAMPLES=-1 = all)
  if [ "$SAMPLES" -lt 0 ]; then FIG6_SAMPLES="$SAMPLES"; else FIG6_SAMPLES="$((SAMPLES > 8 ? SAMPLES : 8))"; fi
  [ "$PLOT_ONLY" = "1" ] || python "$D/compute_mismatch.py" "${COMMON[@]}" \
        --budget "$BUDGET" --samples-per-dataset "$FIG6_SAMPLES" --decode-steps 24
  python "$D/plot_mismatch.py"
fi

# =============================================================================
# Figure 7 — retained-token semantic composition
# =============================================================================
if want 7; then
  echo "### Figure 7: retained-token composition"
  D="$FIG_DIR/fig7_retained_token_composition"
  [ "$PLOT_ONLY" = "1" ] || python "$D/compute_composition.py" "${COMMON[@]}" --budget "$BUDGET"
  python "$D/plot_composition.py"
fi

# =============================================================================
# Figure 8 — layer-wise / head-wise analysis
# =============================================================================
if want 8; then
  echo "### Figure 8: layer / head analysis"
  D="$FIG_DIR/fig8_layer_head_analysis"
  [ "$PLOT_ONLY" = "1" ] || python "$D/compute_layer_head.py" "${COMMON[@]}" \
        --budget "$BUDGET" --decode-steps 24
  python "$D/plot_layer_head.py"          # add --heatmap mismatch for the mismatch heatmap
fi

# =============================================================================
# Figure 9 — budget-equivalence / memory-saving curves
#   (fig9 sweeps its own budget list and uses --score-budget to configure the model,
#    so it does NOT take --budget)
# =============================================================================
if want 9; then
  echo "### Figure 9: budget-equivalence"
  D="$FIG_DIR/fig9_budget_equivalence"
  [ "$PLOT_ONLY" = "1" ] || python "$D/compute_budget.py" \
        --benchmark "$BENCHMARK" --config "$CONFIG_FILE" --dataset "$DATASET" \
        --samples-per-dataset "$SAMPLES" --budgets 64 128 256 512 1024 --score-budget 512 --decode-steps 24
  python "$D/plot_budget.py"
fi

# =============================================================================
# Figure 10 — particle trajectory disagreement D_m (the term that replaced the
#   entropy penalty in the confidence score R_m, Eq. 7)
#
#   Panels A/B relate each particle's D_m and its predictive entropy to that
#   particle's ORACLE utility on the SAME particles, so the two signals can be
#   compared directly; panel C ranks every signal by Spearman correlation with
#   utility; panel D re-weights the identical particles under uniform / log-prob /
#   log-prob-entropy / log-prob-disagreement weights and compares the resulting
#   coverage of Q_future (weights are the only thing that differs -- no extra
#   test-time compute enters the comparison).
#
#   Needs M >= 2 to be meaningful (D_m is undefined for a single particle), hence
#   --num-particles 8, matching figures 4 and 5.
# =============================================================================
if want 10; then
  echo "### Figure 10: particle trajectory divergence vs. utility"
  D="$FIG_DIR/fig10_particle_divergence"
  [ "$PLOT_ONLY" = "1" ] || python "$D/compute_divergence.py" "${COMMON[@]}" \
        --budget "$BUDGET" --num-particles 8 --decode-steps 32
  python "$D/plot_divergence.py"          # add --utility align for the coverage-based utility
fi

echo "=============================================================="
echo " Done. Figures + .npz data written under: $FIG_DIR/figN_*/"
echo "=============================================================="

