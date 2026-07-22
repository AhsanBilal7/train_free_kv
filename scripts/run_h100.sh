#!/bin/bash
#SBATCH -p sooner_gpu_test_h100
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
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$REPO_ROOT/config/default.conf}"

_load_conf() {
    local file="$1"
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue   # skip comments
        [[ "$line" =~ ^[[:space:]]*$ ]] && continue   # skip blank lines
        if [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=[[:space:]]*(.*) ]]; then
            local key="${BASH_REMATCH[1]}"
            local value="${BASH_REMATCH[2]}"
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

source /scratch/ahsan/venvs/kv44/bin/activate

mkdir -p /home/ahsan/LLM-AnDPro/logs

export CUDA_VISIBLE_DEVICES=0
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

# --- Path Configuration (defaults pull straight from the Hugging Face Hub) ---
export MODEL_PATH="${MODEL_PATH:-mistralai/Mistral-7B-Instruct-v0.2}"
export DATASET_PATH="${DATASET_PATH:-THUDM/LongBench}"

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
# LongBench dataset subset(s) to evaluate on.
# Full English set (Chinese-only tasks omitted — Mistral-7B is not tuned for
# Chinese):
#   qasper narrativeqa multifieldqa_en hotpotqa 2wikimqa musique trec triviaqa
#   samsum gov_report qmsum multi_news passage_count passage_retrieval_en
#   lcc repobench-p
# =============================================================================
export DATASET_NAME_LIST="${DATASET_NAME_LIST:-qasper narrativeqa multifieldqa_en hotpotqa 2wikimqa musique trec triviaqa samsum gov_report qmsum multi_news passage_count passage_retrieval_en lcc repobench-p}"
echo "Running with DATASET_NAME_LIST: $DATASET_NAME_LIST"

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

cd experiments/LongBench
bash run.sh
# Example CLI overrides (uncomment one to use):
# MODE=causal_lookahead BUDGET="128 256" NUM_PARTICLES="4" LOOKAHEAD_LEN="8" DATASET_NAME_LIST="qasper trec" bash scripts/run_h100.sh
# MODE=causal_lookahead LAMBDA_LOOKAHEAD="0.0" BUDGET="128 256" DATASET_NAME_LIST="qasper trec" bash scripts/run_h100.sh  # window-only (no lookahead)



