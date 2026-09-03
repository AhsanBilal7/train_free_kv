#!/bin/bash
# =============================================================================
# ACES (TAMU HPRC) — LORE-KV / TLC-KV analysis-figure driver.
#
# Scheduler header mirrors scripts/run_h100_aces.sh so a figure job and an
# experiment job ask for exactly the same resources.
# =============================================================================
#SBATCH --job-name=lorekv-figures
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=/home/u.ab353540/train_free_kv/logs/out_%x_%j.log
#SBATCH --error=/home/u.ab353540/train_free_kv/logs/err_%x_%j.log
# Same ACES association run_h100_aces.sh pins; keeps sbatch from warning
# "Job submitted with no account set".
#SBATCH --account=158535403388
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
#   sbatch scripts/run_figures_aces.sh                    # all figures, LongBench single-doc QA
#   bash   scripts/run_figures_aces.sh                    # same, inside an allocation
#   FIGURES="1 2 6" bash scripts/run_figures_aces.sh      # only figures 1, 2, 6
#   BENCHMARK=ruler DATASET=8192 bash scripts/run_figures_aces.sh
#   BUDGET=256 SAMPLES=5 CONFIG_FILE=config/lorekv-dis-unif-topk.conf bash scripts/run_figures_aces.sh
#   PLOT_ONLY=1 bash scripts/run_figures_aces.sh          # re-draw from existing .npz (no GPU needed)
#
# Interactive allocation with the same resources as the header:
#   srun --time=48:00:00 --partition=gpu --gres=gpu:1 --ntasks=1 --mem=80G \
#        --cpus-per-task=8 --ntasks-per-node=1 --pty bash -i
#   cd ~/train_free_kv && FIGURES=10 bash scripts/run_figures_aces.sh
#
# NOTE on --gres: `gpu:1` (matching the srun above) can land on one of the
# partition's 24 GB A30s, which cannot hold a 7B/8B model plus a long-context KV
# cache. On a CUDA OOM during prefill, check `nvidia-smi -L` in the log and pin
# the card: sbatch --gres=gpu:h100:1 scripts/run_figures_aces.sh
# (or lower FIG_MAX_ANALYSIS_TOKENS below).

module purge
module load WebProxy
module load GCC/11.3.0
module load CUDA/12.1.1


set -euo pipefail

# Resolve the repo root. sbatch copies this script into a spool dir (e.g.
# /var/slurm/spool/slurmd/job<ID>/slurm_script), so BASH_SOURCE points OUTSIDE the
# checkout under SLURM and .env would never be found -- which leaves VENV_PATH
# unset and kills the job on the `:?` below. Prefer the submit dir, exactly as
# scripts/run_h100_aces.sh does.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}}"
if [[ ! -d "$REPO_ROOT/experiments" ]]; then
    REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
if [[ ! -d "$REPO_ROOT/experiments" ]]; then
    echo "ERROR: no experiments/ under REPO_ROOT=$REPO_ROOT" >&2
    echo "       submit from the repo root, or pass REPO_ROOT=/path/to/train_free_kv" >&2
    exit 1
fi
export REPO_ROOT

# --- ACES scratch root --------------------------------------------------------
# .env derives every cache path from $SCRATCH; ACES uses the /scratch/user/$USER
# layout and exports SCRATCH in the login profile, but a batch job can start with
# a clean environment, so re-derive it here BEFORE .env is sourced.
if [[ -z "${SCRATCH:-}" && -d "/scratch/user/$USER" ]]; then
    export SCRATCH="/scratch/user/$USER"
fi

# --- Load central path/env configuration (.env) ---
set -a
[[ -f "$REPO_ROOT/.env" ]] && source "$REPO_ROOT/.env"
set +a


# =============================================================================
# Environment exports (identical policy to scripts/run_h100.sh: nothing under $HOME)
# =============================================================================
source "${VENV_PATH:?VENV_PATH not set — check .env}/bin/activate"

mkdir -p "$REPO_ROOT/logs"

# Redirect every cache/tmp dir to scratch. .env normally exports all of these
# (derived from $SCRATCH), so the fallbacks below only fire if .env is missing --
# they key off $SCRATCH, NOT a hardcoded /scratch/$USER, because on ACES the
# layout is /scratch/user/$USER and mkdir -p on the wrong root aborts the run.
: "${SCRATCH:=/scratch/user/$USER}"
export TMPDIR="${TMPDIR:-$SCRATCH/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRATCH/pip_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH/.cache}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$SCRATCH/torch_extensions}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$SCRATCH/.nv/ComputeCache}"
# Crash-safe per-sample figure checkpoints (common.scratch_ckpt_dir reads FIG_CKPT_DIR).
export FIG_CKPT_DIR="${FIG_CKPT_DIR:-$SCRATCH/lorekv_fig_ckpt}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH" "$FIG_CKPT_DIR"

# Hugging Face caches on scratch.
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HOME"

# Matplotlib config/cache dir (headless Agg backend is set inside the plot scripts).
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRATCH/.cache/matplotlib}"
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
export CONFIG_FILE="${CONFIG_FILE:-$REPO_ROOT/config/lorekv-ent-adap-topk.conf}"
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


# for f in 1 2 3 4 5 6 7 8 9 10 ; do echo "===== Running Figure $f ====="; FIGURES="$f" SAMPLES=20 BENCHMARK=ruler bash scripts/run_figures_aces.sh; status=$?; echo "===== Figure $f finished with exit code $status ====="; done


# for f in 4 ; do echo "===== Running Figure $f ====="; FIGURES="$f" BENCHMARK=longbench bash scripts/run_figures_aces.sh; status=$?; echo "===== Figure $f finished with exit code $status ====="; done



# for cfg in config/lorekv-ent-adap-div.conf config/lorekv-ent-adap-topk.conf config/lorekv-ent-unif-topk.conf config/lorekv-logp-adap-div.conf config/lorekv-logp-adap-topk.conf config/lorekv-logp-unif-topk.conf; do echo "===== Running $cfg ====="; EXPERIMENT=RULER CONFIG_FILE="$cfg" BUDGET="128 256 512 1024" bash scripts/run_h100_aces.sh; done




# for f in 10 ; do echo "===== Running Figure $f ====="; FIGURES="$f" SAMPLES=30 CONFIG_FILE=config/lorekv-dis-adap-topk.conf BUDGET=128 BENCHMARK=ruler bash scripts/run_figures_aces.sh; status=$?; echo "===== Figure $f finished with exit code $status ====="; done


# for config in config/lorekv-ent-unif-topk.conf; do echo "===== Running config: $config ====="; FIGURES=10 SAMPLES=30 CONFIG_FILE="$config" BUDGET=128 BENCHMARK=ruler bash scripts/run_figures_aces.sh; status=$?; echo "===== Config $config finished with exit code $status ====="; done


# NOTE: the old tlckv-m1-* / tlckv-*-mx-* configs only differed in NUM_PARTICLES, which
# is sweepable from the environment, so they have no dedicated config any more:
#   NUM_PARTICLES=1       CONFIG_FILE=config/lorekv-ent-unif-topk.conf ...   (old m1)
#   NUM_PARTICLES="2 4 8" CONFIG_FILE=config/lorekv-ent-unif-topk.conf ...   (old mx)

# /////////////////////////////////

# for config in config/lorekv-dis-adap-topk.conf config/lorekv-dis-adap-topk-llama.conf config/lorekv-dis-unif-topk.conf config/lorekv-dis-unif-topk-llama.conf config/lorekv-dis-adap-div-llama.conf; do echo "===== Running config: $config ====="; FIGURES=10 SAMPLES=-1 CONFIG_FILE="$config" BUDGET=128 BENCHMARK=ruler bash scripts/run_figures_aces.sh; status=$?; echo "===== Config $config finished with exit code $status ====="; done

# for config in config/lorekv-dis-adap-topk.conf config/lorekv-dis-adap-topk-llama.conf config/lorekv-dis-unif-topk.conf config/lorekv-dis-unif-topk-llama.conf config/lorekv-dis-adap-div-llama.conf; do echo "===== Running config: $config ====="; FIGURES=1 SAMPLES=-1 CONFIG_FILE="$config" BUDGET=128 BENCHMARK=ruler bash scripts/run_figures_aces.sh; status=$?; echo "===== Config $config finished with exit code $status ====="; done

# for config in config/lorekv-dis-adap-topk.conf config/lorekv-dis-adap-topk-llama.conf config/lorekv-dis-unif-topk.conf config/lorekv-dis-unif-topk-llama.conf config/lorekv-dis-adap-div-llama.conf; do echo "===== Running config: $config ====="; FIGURES=10 SAMPLES=-1 CONFIG_FILE="$config" BUDGET=128 BENCHMARK=longbench bash scripts/run_figures_aces.sh; status=$?; echo "===== Config $config finished with exit code $status ====="; done

# for config in config/lorekv-dis-adap-topk.conf config/lorekv-dis-adap-topk-llama.conf config/lorekv-dis-unif-topk.conf config/lorekv-dis-unif-topk-llama.conf config/lorekv-dis-adap-div-llama.conf; do echo "===== Running config: $config ====="; FIGURES=1 SAMPLES=-1 CONFIG_FILE="$config" BUDGET=128 BENCHMARK=longbench bash scripts/run_figures_aces.sh; status=$?; echo "===== Config $config finished with exit code $status ====="; done



