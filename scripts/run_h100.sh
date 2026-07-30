#!/bin/bash
#SBATCH -p sooner_gpu_test_h100_dual
#SBATCH --gres=gpu:1
#SBATCH --job-name=tlckv-m1-adap-topk.conf
#SBATCH --output=/home/ahsan/LLM-AnDPro/logs/out_andpro_h100_%j.log
#SBATCH --error=/home/ahsan/LLM-AnDPro/logs/err_andpro_h100_%j.log
#SBATCH --time=2-00:00:00

# CONFIG_FILE=config/tlckv-m1-adap-topk.conf sbatch scripts/run_h100.sh

set -euo pipefail

# Load experiment config (.conf format: KEY = VALUE, # comments, blank lines ignored).
# CLI env-var overrides always win over the config file.
# Override the config path with: CONFIG_FILE=config/my_sweep.conf bash scripts/run_h100.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
# Under SLURM, sbatch copies this script into a spool dir (e.g.
# /var/slurm/spool/slurmd/job<ID>/slurm_script), so BASH_SOURCE-based detection
# points outside the repo. Fall back to the submit dir when the detected root
# lacks the experiments/ tree.
if [[ ! -d "$REPO_ROOT/experiments" && -n "$SLURM_SUBMIT_DIR" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
fi
CONFIG_FILE="${CONFIG_FILE:-$REPO_ROOT/config/default.conf}"

# --- Load central path/env configuration (.env) ---
# All machine-specific paths (venv, scratch caches, HF cache, log dir, model)
# live in $REPO_ROOT/.env. Auto-export everything it defines.
set -a
[[ -f "$REPO_ROOT/.env" ]] && source "$REPO_ROOT/.env"
set +a

_load_conf() {
    local file="$1"
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue   # skip comments
        [[ "$line" =~ ^[[:space:]]*$ ]] && continue   # skip blank lines
        if [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=[[:space:]]*(.*) ]]; then
            local key="${BASH_REMATCH[1]}"
            local value="${BASH_REMATCH[2]}"
            # Strip inline comments (`KEY = 0.0   # entropy penalty OFF`).
            # Values here are numbers, paths, or space-separated sweep lists, so
            # a '#' preceded by whitespace always starts a comment. Without this,
            # the comment words leak into the value and each run.sh sweep loop
            # (`for lambda_h in ${LAMBDA_H}`) iterates over them as parameters.
            value="${value%%[[:space:]]#*}"
            [[ "$value" == "#"* ]] && value=""
            value="${value%"${value##*[![:space:]]}"}"  # trim trailing whitespace
            # Only set if the variable is not already in the environment (CLI wins)
            [[ -z "${!key+x}" ]] && export "$key"="$value"
        fi
    done < "$file"
}

if [[ -f "$CONFIG_FILE" ]]; then
    echo "Loading config: $CONFIG_FILE"
    _load_conf "$CONFIG_FILE"
else
    echo "WARNING: Config file not found: $CONFIG_FILE (falling back to inline defaults)"
fi

source "${VENV_PATH:?VENV_PATH not set — check .env}/bin/activate"

mkdir -p "$LOG_DIR"

# export CUDA_VISIBLE_DEVICES=0
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

if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
    if [ -n "${SLURM_JOB_GPUS:-}" ]; then
        export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
        echo "CUDA_VISIBLE_DEVICES not set, using SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
    else
        echo "CUDA_VISIBLE_DEVICES and SLURM_JOB_GPUS are both unset. Defaulting to 0."
        export CUDA_VISIBLE_DEVICES=0
    fi
else
    echo "CUDA_VISIBLE_DEVICES is set: ${CUDA_VISIBLE_DEVICES}"
fi

nvidia-smi -L || true
export GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
echo "Using GPU_ID=$GPU_ID"

# --- Experiment selection ---
# Which benchmark under experiments/ to run. Switch benchmarks by changing this
# one value (or `EXPERIMENT=RULER bash scripts/run_h100.sh`):
#   LongBench  -> experiments/LongBench
#   RULER      -> experiments/RULER
#   NIAH       -> experiments/NIAH  (conventional single-needle Needle-In-A-Haystack)
export EXPERIMENT="${EXPERIMENT:-LongBench}"
echo "Running experiment: $EXPERIMENT"

# --- Path Configuration (defaults pull straight from the Hugging Face Hub) ---
# NOTE: DATASET_PATH is now dataset-specific and lives in each experiment's
# run.sh (experiments/<EXPERIMENT>/run.sh). It is only read here if the user
# exports it explicitly (which then overrides the experiment default).
export MODEL_PATH="${MODEL_PATH:-mistralai/Mistral-7B-Instruct-v0.2}"

# =============================================================================
# KV eviction mode (§3 of the paper)
#
#   andpro              - Original AnDPro: scores chunks by projecting the
#                         attention-weighted value onto the pre-eviction output
#                         direction (Geng et al., 2026).
#
#   causal_forward_only - Training-Free Causal AnDPro: output-projection-aware
#                         leave-one-out deletion score (Eq. 13-14) evaluated
#                         only under the prompt observation-window queries.
#                         No lookahead; uses LAMBDA_RED for optional redundancy
#                         penalty in greedy chunk selection.
#
#   causal_lookahead    - TLC-KV (this paper): full two-level method.
#                         Generates M short lookahead continuations (Eq. 5),
#                         weights them by model confidence (Eq. 7-8), combines
#                         window + lookahead queries in the counterfactual
#                         output-projection score (Eq. 13-14), aggregates
#                         robustly (Eq. 15-17), allocates head-wise budgets
#                         (Eq. 19-20), and applies optional diversity-aware
#                         greedy selection (Eq. 21).
#
# Sweepable: MODE="andpro causal_forward_only causal_lookahead"
# =============================================================================
export MODE="${MODE:-causal_lookahead}"
echo "Running with MODE(s): $MODE"

# =============================================================================
# KV cache budget B: maximum number of KV pairs retained per head per layer.
#
# This is the shared budget parameter across all modes.
# Sweepable: BUDGET="128 256 512 1024"
# =============================================================================
export BUDGET="${BUDGET:-1024 256 512 128}"
echo "Running with BUDGET(s): $BUDGET"

# =============================================================================
# Dataset subset(s) to evaluate on — dataset-specific, so the DEFAULT lives in
# each experiment's run.sh (experiments/<EXPERIMENT>/run.sh):
#   LongBench -> sub-tasks: qasper narrativeqa multifieldqa_en hotpotqa 2wikimqa
#                musique trec triviaqa samsum gov_report qmsum multi_news
#                passage_count passage_retrieval_en lcc repobench-p
#   RULER     -> sequence-length configs: 4096 8192 16384
#   NIAH      -> target context lengths (tokens): 4096 8192 16384
#                (needle depths are a separate sweep: set NIAH_DEPTHS)
# Set it here (or on the CLI) ONLY to override that per-experiment default, e.g.
#   DATASET_NAME_LIST="qasper trec"  (LongBench)  or  "4096 8192"  (RULER/NIAH)
# Left empty => the experiment's run.sh chooses its own full default set.
# =============================================================================
export DATASET_NAME_LIST="${DATASET_NAME_LIST:-}"
echo "Running with DATASET_NAME_LIST: ${DATASET_NAME_LIST:-<per-experiment default>}"

# =============================================================================
# causal_forward_only — redundancy penalty (ignored by other modes)
#
# lambda_red: weight of the redundancy penalty in the greedy chunk-selection
#   score: S_final(i) = S_causal(i) - LAMBDA_RED * max_j Red(i,j).
#
# To DISABLE the redundancy penalty and revert to plain top-K selection:
#   LAMBDA_RED = 0.0
#
# Sweepable: LAMBDA_RED="0.0 0.3 0.5"
# =============================================================================
export LAMBDA_RED="${LAMBDA_RED:-0.5}"
echo "Running with LAMBDA_RED(s): $LAMBDA_RED"

# =============================================================================
# causal_lookahead — test-time lookahead particle generation (Eq. 4-8)
#   (these parameters are ignored by andpro and causal_forward_only)
#
# NUM_PARTICLES (M): number of short lookahead continuations generated from the
#   frozen model before final decoding (Eq. 5). All M particles are decoded in
#   parallel as a single batched continuation, so wall-clock overhead is that of
#   one length-T decode, independent of M (up to batching limits). Cost is paid
#   once before final generation and is amortized over the full output.
#
# To DISABLE lookahead entirely and use only the prompt observation-window
#   queries for scoring (pure prompt-window output-perturbation method, §3.11):
#     Option A — set LAMBDA_LOOKAHEAD = 0.0  (skips the lookahead term in
#                Eq. 18 without generating any particles; fastest).
#     Option B — set NUM_PARTICLES = 0       (no particles are sampled; the
#                lookahead loop body never executes).
#   Both options produce identical scores to the prompt-window-only baseline.
#
# Sweepable: NUM_PARTICLES="2 4 8"
# =============================================================================
export NUM_PARTICLES="${NUM_PARTICLES:-4}"

# LOOKAHEAD_LEN (T): number of tokens in each lookahead continuation (Eq. 5).
# Longer continuations expose more future query directions but increase the
# one-time generation cost proportionally.
# Sweepable: LOOKAHEAD_LEN="4 8 16"
export LOOKAHEAD_LEN="${LOOKAHEAD_LEN:-8}"

# LAMBDA_LOOKAHEAD (lambda_L): weight of the particle-weighted lookahead score
#   relative to the prompt observation-window score (Eq. 18):
#     S_i = rho(E_i,W) + lambda_L * sum_m [ omega_m * rho(E_i,m) ]
#   lambda_L = 0.0 → pure prompt-window scoring (no lookahead contribution,
#              fastest, equivalent to causal_forward_only without redundancy).
#   lambda_L > 0  → future-query-aware scoring; larger values weight the
#              lookahead queries more heavily than the window queries.
#
# To USE ONLY window queries for scoring (test-time compute = 0 lookahead):
#   LAMBDA_LOOKAHEAD = 0.0
# To USE ONLY lookahead queries (no window contribution): not recommended;
#   use a large LAMBDA_LOOKAHEAD (e.g. 10.0) to strongly dominate the window.
# Default balanced setting: LAMBDA_LOOKAHEAD = 1.0
#
# Sweepable: LAMBDA_LOOKAHEAD="0.0 0.5 1.0 2.0"
export LAMBDA_LOOKAHEAD="${LAMBDA_LOOKAHEAD:-1.0}"

# LOOKAHEAD_TEMPERATURE: softmax temperature used only when sampling lookahead
#   particles. Higher values → more diverse (but lower-probability) particles.
#   Does NOT affect the main (final) answer generation, which always uses
#   greedy decoding. Has no effect when NUM_PARTICLES = 0 or LAMBDA_LOOKAHEAD = 0.
# Sweepable: LOOKAHEAD_TEMPERATURE="0.5 1.0 1.5"
export LOOKAHEAD_TEMPERATURE="${LOOKAHEAD_TEMPERATURE:-1.0}"

echo "Running with (causal_lookahead) NUM_PARTICLES=$NUM_PARTICLES LOOKAHEAD_LEN=$LOOKAHEAD_LEN LAMBDA_LOOKAHEAD=$LAMBDA_LOOKAHEAD"

# =============================================================================
# causal_lookahead — particle confidence weighting (Eq. 7-8)
#   (ignored by andpro and causal_forward_only)
#
# LAMBDA_H (lambda_H): entropy penalty weight in the per-particle confidence
#   score R_m (Eq. 7):
#     R_m = (1/T) * sum_t log p(y_t^m | X, y_{<t}^m)
#           - lambda_H * (1/T) * sum_t H( p(· | X, y_{<t}^m) )
#   A particle with high predictive entropy is penalized, reducing its weight
#   omega_m (Eq. 8) even if its average log-probability is high.
#
# To DISABLE the entropy penalty (use raw average log-probability only):
#   LAMBDA_H = 0.0
#
# Sweepable: LAMBDA_H="0.0 0.1 0.3"
# =============================================================================
export LAMBDA_H="${LAMBDA_H:-0.0}"

# PARTICLE_ETA (eta): inverse-temperature (sharpness) of the particle-weight
#   softmax (Eq. 8):
#     omega_m = exp(eta * R_m) / sum_r exp(eta * R_r)
#   eta → 0  → uniform weights (all particles contribute equally regardless of
#              confidence; confidence weighting is effectively disabled).
#   eta → ∞  → hard selection of the single most-confident particle.
#   Soft weighting (finite eta > 0) is preferred so no single particle dominates.
# Sweepable: PARTICLE_ETA="0.5 1.0 2.0"
export PARTICLE_ETA="${PARTICLE_ETA:-1.0}"

# =============================================================================
# causal_lookahead — robust aggregation over prompt and lookahead queries (Eq. 15-17)
#   (ignored by andpro and causal_forward_only)
#
# For a given token i and query source (window or particle m), the per-head
# perturbation scores E_{i,W} and E_{i,m} are aggregated robustly:
#   rho(E) = alpha * TopAvg_beta(E) + (1 - alpha) * Mean(E)   (Eq. 17)
# where TopAvg_beta keeps the top ceil(beta * |E|) scores (Eq. 16).
#
# ROBUST_ALPHA (alpha): balance between the worst-case-sensitive top-beta
#   average and the average-case mean (Eq. 17).
#   alpha = 0.0 → pure mean over all queries (ignores rare important queries).
#   alpha = 1.0 → pure top-beta average (worst-case / robustness focused).
# Sweepable: ROBUST_ALPHA="0.0 0.5 1.0"
# =============================================================================
export ROBUST_ALPHA="${ROBUST_ALPHA:-0.5}"

# ROBUST_TOPK_BETA (beta): fraction of queries used in the robust top-beta
#   average (Eq. 16). K_beta = ceil(beta * |E|).
#   beta = 1.0 → all queries kept → TopAvg_beta = Mean → rho = Mean regardless
#               of alpha (equivalent to disabling the robust aggregation).
#   beta = 0.5 → top 50% of per-query scores kept.
# Sweepable: ROBUST_TOPK_BETA="0.25 0.5 0.75 1.0"
export ROBUST_TOPK_BETA="${ROBUST_TOPK_BETA:-0.5}"

# =============================================================================
# causal_lookahead — diversity-aware greedy selection (Eq. 21)
#   (ignored by andpro and causal_forward_only)
#
# Greedy selection penalizes chunks whose key or value representations are
# already well covered by previously selected chunks:
#   i* = argmax_i [ S_i - lambda_k * max_j cos(k_i, k_j)
#                        - lambda_v * max_j cos(v_i, v_j) ]
#
# To DISABLE diversity-aware selection and use plain top-score ranking:
#   LAMBDA_K = 0.0  AND  LAMBDA_V = 0.0
#   (both penalties must be zero; setting only one to zero keeps the other active)
#
# Non-zero values are most useful when the context contains many near-duplicate
# passages, but can wrongly penalize repeated evidence that is genuinely
# important (e.g., the same entity repeated across supporting passages in
# multi-document QA). Keep values small so redundancy only breaks ties.
#
# Sweepable: LAMBDA_K="0.0 0.1 0.5"
# =============================================================================
export LAMBDA_K="${LAMBDA_K:-0.0}"

# lambda_v: value-similarity cosine diversity penalty in greedy selection (Eq. 21).
# To DISABLE: LAMBDA_V = 0.0  (along with LAMBDA_K = 0.0)
# Sweepable: LAMBDA_V="0.0 0.1 0.5"
export LAMBDA_V="${LAMBDA_V:-0.0}"

# =============================================================================
# causal_lookahead — adaptive head-wise budget allocation (Eq. 19-20)
#   (ignored by andpro and causal_forward_only)
#
# BUDGET_TAU (tau): controls how aggressively the per-head budget B_{l,h}
#   (Eq. 20) shifts toward heads whose candidate deletions produce large
#   future-query perturbation mass E_{l,h} (Eq. 19):
#     B_{l,h} = floor[ B_l * (E_{l,h} + eps)^tau
#                          / sum_{h'} (E_{l,h'} + eps)^tau ]
#   tau = 0.0 → uniform budget across all heads (no adaptive allocation;
#               same total tokens per head as a flat budget would give).
#   tau = 1.0 → linear proportional allocation to perturbation mass.
#   tau > 1.0 → increasingly aggressive concentration on high-impact heads.
# Sweepable: BUDGET_TAU="0.0 0.5 1.0 2.0"
# =============================================================================
export BUDGET_TAU="${BUDGET_TAU:-1.0}"

# =============================================================================
# Analysis figures inside each run (opt-in; causal_lookahead runs only)
#
# When MAKE_FIGURES=1, after each causal_lookahead prediction finishes, run.sh
# also computes + saves every analysis figure's .npz AND .png INTO that run's
# output folder (experiments/<EXP>/pred/<out_name>/), one artifact PER dataset
# plus one COMBINED artifact across all datasets in the run. The figures reuse
# the run's exact hyper-parameters (synthesized into a figure_config.conf).
#
# COST: each (figure x dataset) reloads the model, so this is heavy. Scope it
# with FIG_SAMPLES (samples/dataset), FIG_LIST (which figures), and the run's
# own DATASET_NAME_LIST. The main experiment is unaffected when MAKE_FIGURES=0.
#
# Example:  MAKE_FIGURES=1 FIG_SAMPLES=2 FIG_LIST="1 2 6" \
#           CONFIG_FILE=config/tlckv-pe-m4-adap-topk.conf bash scripts/run_h100.sh
# =============================================================================
export MAKE_FIGURES="${MAKE_FIGURES:-0}"
export FIG_SAMPLES="${FIG_SAMPLES:-2}"
export FIG_LIST="${FIG_LIST:-1 2 3 4 5 6 7 8 9}"
echo "MAKE_FIGURES=$MAKE_FIGURES (FIG_SAMPLES=$FIG_SAMPLES FIG_LIST='$FIG_LIST')"

cd "$REPO_ROOT/experiments/$EXPERIMENT"
bash run.sh
# Example CLI overrides (uncomment one to use):
# EXPERIMENT=RULER MODE=causal_lookahead BUDGET="128 256" DATASET_NAME_LIST="4096 8192" bash scripts/run_h100.sh
# MODE=causal_lookahead BUDGET="128 256" NUM_PARTICLES="4" LOOKAHEAD_LEN="8" DATASET_NAME_LIST="qasper trec" bash scripts/run_h100.sh
# MODE=causal_lookahead LAMBDA_LOOKAHEAD="0.0" BUDGET="128 256" DATASET_NAME_LIST="qasper trec" bash scripts/run_h100.sh  # window-only (no lookahead)



