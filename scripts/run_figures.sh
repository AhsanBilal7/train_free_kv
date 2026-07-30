#!/bin/bash
#SBATCH -p sooner_gpu_test_h100
#SBATCH --gres=gpu:1
#SBATCH --job-name=lorekv-figures
#SBATCH --output=/home/ahsan/LLM-AnDPro/logs/out_figures_%j.log
#SBATCH --error=/home/ahsan/LLM-AnDPro/logs/err_figures_%j.log
#SBATCH --time=1-00:00:00
#
# LORE-KV / TLC-KV analysis-figure driver.
#
# Reproduces the SAME environment exports as scripts/run_h100.sh (venv, scratch
# cache/tmp dirs, Hugging Face cache, GPU selection) and then runs the two-stage
# compute -> plot pipeline for every mechanism figure under
#   experiments/<BENCHMARK>/figure_making_script/figN_*/
#
# Each figure has a compute_*.py (saves a .npz next to itself) and a plot_*.py
# (reads that .npz, writes the .png). This script runs both.
#
# Usage:
#   sbatch scripts/run_figures.sh                         # all figures, LongBench single-doc QA
#   bash   scripts/run_figures.sh                         # same, interactively
#   FIGURES="1 2 6" bash scripts/run_figures.sh           # only figures 1, 2, 6
#   BENCHMARK=ruler DATASET=8192 bash scripts/run_figures.sh
#   BUDGET=256 SAMPLES=5 CONFIG_FILE=config/tlckv-pe-mx-unif-topk.conf bash scripts/run_figures.sh
#   PLOT_ONLY=1 bash scripts/run_figures.sh              # re-draw from existing .npz (no GPU needed)

module purge
module load WebProxy
module load GCC/11.3.0
module load CUDA/12.1.1


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Load central path/env configuration (.env) ---
set -a
[[ -f "$REPO_ROOT/.env" ]] && source "$REPO_ROOT/.env"
set +a


# =============================================================================
# Environment exports (identical policy to scripts/run_h100.sh: nothing under $HOME)
# =============================================================================
source "${VENV_PATH:?VENV_PATH not set — check .env}/bin/activate"

mkdir -p "$REPO_ROOT/logs"

# Redirect every cache/tmp dir to scratch.
export TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/$USER/pip_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/$USER/.cache}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/scratch/$USER/torch_extensions}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-/scratch/$USER/.nv/ComputeCache}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH"

# Hugging Face caches on scratch.
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HOME"

# Matplotlib config/cache dir (headless Agg backend is set inside the plot scripts).
export MPLCONFIGDIR="${MPLCONFIGDIR:-/scratch/$USER/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

# Reduce CUDA fragmentation across long/variable-length samples (the figures free
# each sample before the next, but expandable segments avoids allocator fragmentation
# when a longer dataset follows a shorter one).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Cap the prompt length used for the analysis (middle-truncation) so very long samples
# (e.g. NarrativeQA ~40K) still contribute without OOMing prefill + M cache-clones on a
# shared GPU. 0 = no cap. Raise it if you have the whole GPU to yourself.
export FIG_MAX_ANALYSIS_TOKENS="${FIG_MAX_ANALYSIS_TOKENS:-24000}"

# --- GPU selection (same fallback logic as run_h100.sh) ---
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
    if [ -n "${SLURM_JOB_GPUS:-}" ]; then
        export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
    else
        export CUDA_VISIBLE_DEVICES=0
    fi
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi -L || true

# =============================================================================
# Figure-run configuration (override any of these via env vars / CLI)
# =============================================================================
export BENCHMARK="${BENCHMARK:-longbench}"          # longbench | ruler
export CONFIG_FILE="${CONFIG_FILE:-$REPO_ROOT/config/tlckv-pe-m4-adap-topk.conf}"
export BUDGET="${BUDGET:-128}"                      # tightest budget is most diagnostic
export SAMPLES="${SAMPLES:--1}"                      # samples per dataset/sub-task (-1 = all samples)
export FIGURES="${FIGURES:-1 2 3 4 5 6 7 8 9}"      # which figures to (re)build
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
echo $FIG_CKPT_DIR

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

echo "=============================================================="
echo " Done. Figures + .npz data written under: $FIG_DIR/figN_*/"
echo "=============================================================="


# for f in 3 4 5 9 8 7 2 1; do echo "===== Running Figure $f ====="; FIGURES="$f" BENCHMARK=ruler bash scripts/run_figures.sh; status=$?; echo "===== Figure $f finished with exit code $status ====="; done