#!/bin/bash
# =============================================================================
# ACES (TAMU HPRC) — H100 batch launcher
#
# ACES counterpart of scripts/run_h100.sh (whose #SBATCH header targets OSCER's
# sooner_gpu_test_h100_dual partition). This file carries ONLY the ACES-specific
# scheduler header plus the job's defaults, then hands off to run_h100.sh, which
# stays the single source of truth for .env loading, config parsing, scratch/HF
# cache setup and the experiment dispatch.
#
# Two modes, selected by whether CONFIG_FILE is set:
#
#   CONFIG_FILE UNSET -> sweep the reparametrized (BETA, GAMMA, LAMBDA_D) grid,
#       running config/lorekv-<cell>-<variant>[-llama].conf for every combination of
#       CELLS x VARIANTS x MODELS (all three overridable; defaults give 10 runs).
#
#   CONFIG_FILE SET   -> run exactly that one config and SKIP the sweep. A bare
#       config name works too, so these are equivalent:
#           sbatch --export=ALL,CONFIG_FILE=config/lorekv-typ-unif-topk.conf ...
#           sbatch --export=ALL,CONFIG_FILE=lorekv-typ-unif-topk ...
#
# Usage (from the repo root):
#   sbatch scripts/run_h100_aces.sh
#
# Or inside an interactive allocation with the same resources as the header:
#   srun --time=48:00:00 --partition=gpu --gres=gpu:1 --ntasks=1 --mem=80G \
#        --cpus-per-task=8 --ntasks-per-node=1 --pty bash -i
#   cd ~/train_free_kv && EXPERIMENT=LongBench \
#       CONFIG_FILE=config/lorekv-ent-unif-topk.conf bash scripts/run_h100_aces.sh
#
# Every default below is overridable at submit time (CLI wins over this file,
# which wins over $CONFIG_FILE):
#   sbatch --export=ALL,BUDGET="512 1024" scripts/run_h100_aces.sh
#   sbatch --export=ALL,CONFIG_FILE=lorekv-typ-unif-topk scripts/run_h100_aces.sh
#   sbatch --export=ALL,CELLS="dis typ-dis",MODELS=mistral scripts/run_h100_aces.sh
#   sbatch --export=ALL,VARIANTS="unif-topk adap-topk unif-div adap-div" \
#          scripts/run_h100_aces.sh
#
# ACES GPU partitions (sinfo):
#   gpu        2-00:00:00  h100 x{1,3,4,8} on ac[041,045,049,051,055,096], a30 x2
#   gpu_debug     2:00:00  h100 x2 on ac098, a30 x2  <- quick smoke tests
#
# NOTE on --gres: the header asks for a generic `gpu:1`, matching the interactive
# srun above. The gpu partition also holds 24 GB A30s, and a 7B/8B model in bf16
# plus a long-context KV cache does NOT fit on one. If a job dies with a CUDA OOM
# during prefill, check `nvidia-smi -L` in the log and pin the card instead:
#   #SBATCH --gres=gpu:h100:1        (or: sbatch --gres=gpu:h100:1 scripts/run_h100_aces.sh)
# =============================================================================
#SBATCH --job-name=lorekv-aces
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
# ACES rejects a job with no wall time ("(from job_submit) wall time (--time) is
# not specifed"), so --time above is REQUIRED, not decorative. The account keeps
# sbatch from warning "Job submitted with no account set".
#SBATCH --account=158535403388


module purge
module load WebProxy
module load GCC/11.3.0
module load CUDA/12.1.1

set -euo pipefail

# Resolve the repo root. sbatch copies this script to a spool dir, so
# BASH_SOURCE points outside the checkout under SLURM -- prefer the submit dir.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}}"
if [[ ! -d "$REPO_ROOT/experiments" ]]; then
    echo "ERROR: no experiments/ under REPO_ROOT=$REPO_ROOT" >&2
    echo "       submit from the repo root, or pass REPO_ROOT=/path/to/train_free_kv" >&2
    exit 1
fi
export REPO_ROOT

# --- Job definition (all overridable via --export=ALL,VAR=...) ---------------
export EXPERIMENT="${EXPERIMENT:-LongBench}"
# export MODE="${MODE:-andpro}"
# B=128 only for the reliability grid: Table 3 separates the criterion by 5.05
# points at B=128 and 0.42 at B=1024, so larger budgets cannot resolve a
# weighting rule. Override for a full budget sweep of a single config.
export BUDGET="${BUDGET:-128}"

# Was CONFIG_FILE supplied by the caller? Recorded BEFORE defaulting, because that
# is what selects between the two modes:
#   CONFIG_FILE given  -> run exactly that one config; the CELLS/VARIANTS/MODELS
#                         grid sweep below is SKIPPED entirely.
#   CONFIG_FILE unset  -> sweep the grid (see ACTIVE SWEEP below).
# `:+` not `+`: an empty CONFIG_FILE="" counts as UNSET and sweeps, matching the
# `${CONFIG_FILE:-...}` default on the next line. Otherwise `--export=ALL,CONFIG_FILE=`
# would silently run the default config once instead of doing what it looks like.
_CONFIG_GIVEN="${CONFIG_FILE:+set}"
CONFIG_FILE="${CONFIG_FILE:-config/lorekv-ent-unif-topk.conf}"
# Convenience: accept a bare config name as well as a path, so
# `CONFIG_FILE=lorekv-typ-unif-topk` works like `CONFIG_FILE=config/lorekv-typ-unif-topk.conf`.
if [[ "$CONFIG_FILE" != */* && "$CONFIG_FILE" != *.conf ]]; then
    CONFIG_FILE="config/${CONFIG_FILE}.conf"
fi
# run_h100.sh only checks `-f "$CONFIG_FILE"`, so make a relative path absolute
# against the repo root -- the compute node's cwd is not guaranteed to be there.
[[ "$CONFIG_FILE" != /* ]] && CONFIG_FILE="$REPO_ROOT/$CONFIG_FILE"
export CONFIG_FILE

# --- ACES scratch root --------------------------------------------------------
# .env derives every cache path from $SCRATCH; ACES uses the /scratch/user/$USER
# layout and does export SCRATCH in the login profile, but a batch job can start
# with a clean environment, so re-derive it here before .env is sourced.
if [[ -z "${SCRATCH:-}" && -d "/scratch/user/$USER" ]]; then
    export SCRATCH="/scratch/user/$USER"
fi

echo "=============================================================="
echo "ACES H100 job ${SLURM_JOB_ID:-<interactive>} on $(hostname)"
echo "  REPO_ROOT   = $REPO_ROOT"
echo "  EXPERIMENT  = $EXPERIMENT"
echo "  BUDGET      = $BUDGET"
if [[ -n "$_CONFIG_GIVEN" ]]; then
    echo "  RUN MODE    = single config (grid sweep skipped)"
    echo "  CONFIG_FILE = $CONFIG_FILE"
else
    echo "  RUN MODE    = grid sweep over CELLS x VARIANTS x MODELS"
fi
echo "  SCRATCH     = ${SCRATCH:-<unset>}"
echo "=============================================================="

# --- Single-config mode -------------------------------------------------------
# An explicit CONFIG_FILE means the caller already knows which cell/variant/model
# they want, so hand straight off to the shared launcher and skip the sweep.
#
# The existence check is not optional: run_h100.sh only tests `-f "$CONFIG_FILE"`
# and, on a miss, prints a warning and silently falls back to its INLINE DEFAULTS --
# i.e. a typo would quietly run the WRONG configuration instead of failing.
if [[ -n "$_CONFIG_GIVEN" ]]; then
    if [[ ! -f "$CONFIG_FILE" ]]; then
        _base="$(basename "$CONFIG_FILE")"
        _stem="${_base%.conf}"
        echo "ERROR: config not found: $CONFIG_FILE" >&2
        if [[ -f "$REPO_ROOT/config/$_base" ]]; then
            # Right file, wrong directory -- the overwhelmingly common case is a
            # `conf/` vs `config/` slip. Say the fix instead of just the failure.
            echo "       Did you mean:  CONFIG_FILE=config/$_base" >&2
            echo "                  or:  CONFIG_FILE=$_stem        (a bare name works too)" >&2
            echo "       (configs live in config/, not $(dirname "${CONFIG_FILE#$REPO_ROOT/}")/)" >&2
        else
            # Wrong name: suggest the closest configs. Drop one trailing '-token' at a
            # time until something matches, so `lorekv-ent-adap-tokp` narrows to the
            # `lorekv-ent-adap-*` pair rather than to every config starting `lorekv`.
            # `|| true` is required: a non-matching grep exits 1, and under `set -e` a
            # failing command substitution in an assignment aborts the whole script --
            # which truncated this very message mid-way.
            _near=""
            _probe="$_stem"
            while [[ -n "$_probe" ]]; do
                _near="$(ls "$REPO_ROOT/config" 2>/dev/null | grep -F -- "$_probe" || true)"
                [[ -n "$_near" ]] && break
                [[ "$_probe" != *-* ]] && break
                _probe="${_probe%-*}"
            done
            if [[ -n "$_near" ]]; then
                echo "       Closest configs (matched '$_probe'):" >&2
                printf '         config/%s\n' $(echo "$_near" | head -8) >&2
                [[ "$(echo "$_near" | wc -l)" -gt 8 ]] && echo "         ... ($(echo "$_near" | wc -l) total)" >&2
            else
                echo "       Configs are named lorekv-<cell>-<brule>-<sel>[-llama].conf, e.g." >&2
                echo "         config/lorekv-ent-unif-topk.conf   (the core configuration)" >&2
                echo "       List them with:  ls config/" >&2
            fi
        fi
        exit 1
    fi
    echo "===== Running $CONFIG_FILE ====="
    bash "$REPO_ROOT/scripts/run_h100.sh"
    exit 0
fi




# =============================================================================
# ACTIVE SWEEP -- the reparametrized (BETA, GAMMA, LAMBDA_D) grid
#
# Reached ONLY when no CONFIG_FILE was given; an explicit one exits above.
#
#   R_m = BETA * cbar_m - GAMMA * Hbar_m - LAMBDA_D * zscore_m(D)
#   c_u = log q(y_u|h_u) + H(q(.|h_u)),   E[c_u | h_u] = 0
#
# Config naming:  lorekv-<cell>-<budget>-<selection>[-llama].conf
#   <cell>       reliability coefficients (the axis this grid is about)
#   <budget>     unif = uniform head budgets (tau=0)   adap = adaptive (tau=1)
#   <selection>  topk = plain top-K                    div  = diversity-aware
#
# Hold <budget>-<selection> FIXED and sweep <cell>: that is the single-factor
# reliability comparison. Sweeping the other way prices allocation/selection at a
# fixed reliability rule. unif-topk is the paper's core configuration.
#
# Stage 1 spans the (BETA, GAMMA) plane at LAMBDA_D = 0 and contains BOTH controls the
# old (lambda_H, eta) parametrization could not express: uniform weights (0,0,0) and
# pure typicality (1,0,0). lorekv-logp (1,1,0) and lorekv-ent (1,1.3,0) relabel the
# two published cells exactly, so they are the reproducibility check -- at unif-topk
# they must come back at 40.29/41.39/41.30/41.40 and 40.28/41.38/41.84/42.21.
#
# B=128 ONLY. Table 3 separates the criterion by 5.05 points at B=128 and 0.42 at
# B=1024, so the larger budgets have no resolving power for a weighting rule.
#
# Stage 2 (dis, typ-dis) is GATED on test S4 -- run
# experiments/LongBench/figure_making_script/reparam_validation/ first, per benchmark.
# Add them to CELLS only once S4 comes back non-zero and sign-consistent.
# =============================================================================
# All three axes are space-separated strings so they can be overridden at submit
# time:  sbatch --export=ALL,VARIANTS="unif-topk adap-div",MODELS=mistral ...
#
# CELLS defaults to Stage 1. Add "dis typ-dis" only after test S4 passes.
read -ra CELLS    <<< "${CELLS:-uniform typ logp ent ent-only}"
# Budget x selection variants each cell runs under. Default is the core
# configuration only; the full 2x2 cross is "unif-topk adap-topk unif-div adap-div".
read -ra VARIANTS <<< "${VARIANTS:-unif-topk}"
read -ra MODELS   <<< "${MODELS:-mistral llama}"

echo "sweep: cells=[${CELLS[*]}] variants=[${VARIANTS[*]}] models=[${MODELS[*]}]"

for cell in "${CELLS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        for model in "${MODELS[@]}"; do
            suffix=""
            [[ "$model" == "llama" ]] && suffix="-llama"
            cfg="config/lorekv-${cell}-${variant}${suffix}.conf"
            if [[ ! -f "$REPO_ROOT/$cfg" ]]; then
                echo "SKIP (missing): $cfg" >&2
                continue
            fi
            echo "===== Running $cfg ====="
            EXPERIMENT="$EXPERIMENT" BUDGET="${BUDGET:-128}" \
                CONFIG_FILE="$REPO_ROOT/$cfg" \
                bash "$REPO_ROOT/scripts/run_h100.sh"
        done
    done
done
# =============================================================================


# =============================================================================
# Reference command sets (uncomment one; each is a complete sweep)
# =============================================================================
#
# The loops below call scripts/run_h100.sh DIRECTLY rather than re-entering this
# wrapper once per config: inside an interactive allocation the wrapper's job is
# already done (modules loaded, SCRATCH derived, REPO_ROOT resolved), so calling
# it again per config just repeats `module purge`/`module load` N times.
#
# Run this ONCE first, from the repo root, in the interactive shell:
#
#   module purge && module load WebProxy GCC/11.3.0 CUDA/12.1.1
#   export REPO_ROOT="$PWD" SCRATCH="${SCRATCH:-/scratch/user/$USER}"
#
# CONFIG_FILE is passed as "$REPO_ROOT/$cfg" (absolute) on purpose: run_h100.sh
# only tests `-f "$CONFIG_FILE"` against the CURRENT directory and, on a miss,
# prints a warning and silently falls back to its inline defaults -- i.e. a
# relative path from the wrong cwd runs the WRONG configuration instead of
# failing. Each `bash run_h100.sh` is its own process, so nothing a config
# exports leaks into the next iteration.
# -----------------------------------------------------------------------------

# --- Stage 1: (BETA, GAMMA) plane at LAMBDA_D = 0, LongBench, Mistral ---
# for cell in uniform typ logp ent ent-only; do cfg="lorekv-$cell-unif-topk"; echo "===== Running $cfg ====="; EXPERIMENT=LongBench CONFIG_FILE="$REPO_ROOT/config/$cfg.conf" BUDGET=128 bash "$REPO_ROOT/scripts/run_h100.sh"; done

# --- Stage 1 on RULER ---
# for cell in uniform typ logp ent ent-only; do cfg="lorekv-$cell-unif-topk"; echo "===== Running $cfg ====="; EXPERIMENT=RULER CONFIG_FILE="$REPO_ROOT/config/$cfg.conf" BUDGET=128 bash "$REPO_ROOT/scripts/run_h100.sh"; done

# --- Stage 1 on Llama ---
# for cell in uniform typ logp ent ent-only; do cfg="lorekv-$cell-unif-topk-llama"; echo "===== Running $cfg ====="; EXPERIMENT=LongBench CONFIG_FILE="$REPO_ROOT/config/$cfg.conf" BUDGET=128 bash "$REPO_ROOT/scripts/run_h100.sh"; done

# --- Three seeds. The gaps at stake are 0.01 (Table 4) and 0.04 (Table 11), and the
#     existing M-sweep is non-monotone (M=2 sits BELOW M=1), so the noise floor is
#     unknown and may exceed them. MODEL_TAG keeps the seeds in separate pred/ dirs. ---
# for seed in 0 1 2; do for cell in uniform typ logp ent; do cfg="lorekv-$cell-unif-topk"; echo "===== $cfg seed=$seed ====="; EXPERIMENT=LongBench CONFIG_FILE="$REPO_ROOT/config/$cfg.conf" BUDGET=128 SEED=$seed PYTHONHASHSEED=$seed MODEL_TAG="Mistral-7B-Instruct-v0.2-s$seed" bash "$REPO_ROOT/scripts/run_h100.sh"; done; done

# --- Stage 2: the LAMBDA_D probe. GATED ON TEST S4 -- do not run this before the
#     semipartial correlation of D_m with utility, controlling for (cbar_m, Hbar_m),
#     comes back non-zero AND sign-consistent PER BENCHMARK. Figure 10 already shows
#     -Dhat_m at rho = -0.26 on LongBench and +0.03 on RULER, so a pooled number would
#     hide a sign flip. lorekv-typ-dis (1,0,0.3) is disagreement INSTEAD of entropy;
#     lorekv-dis (1,1,0.3) is what the old /D actually was -- ON TOP of a full entropy
#     penalty. Calibrate LAMBDA_D so sd_m(LAMBDA_D*Dhat) matches sd_m(GAMMA*Hbar). ---
# for cell in dis typ-dis; do cfg="lorekv-$cell-unif-topk"; echo "===== Running $cfg ====="; EXPERIMENT=LongBench CONFIG_FILE="$REPO_ROOT/config/$cfg.conf" BUDGET=128 bash "$REPO_ROOT/scripts/run_h100.sh"; done

# --- LAMBDA_D sweep on one config (out_name carries _lD_<value>, so runs never
#     collide; LAMBDA_D=0.0 reproduces the corresponding lambda_D-free cell exactly) ---
# EXPERIMENT=LongBench CONFIG_FILE="$REPO_ROOT/config/lorekv-typ-dis-unif-topk.conf" LAMBDA_D="0.0 0.1 0.3 1.0" BUDGET=128 bash "$REPO_ROOT/scripts/run_h100.sh"

# --- Validation. Run BEFORE reading any reliability number. V1 comes first: a
#     positive mean cbar_m means the sampling law is not the centering law, and
#     nothing downstream is interpretable until that is fixed and re-captured.
#     Run LongBench and RULER SEPARATELY -- never pooled. ---
# cd "$REPO_ROOT/experiments/LongBench/figure_making_script/reparam_validation"
# python compute_validation.py --config "$REPO_ROOT/config/lorekv-ent-unif-topk.conf" --benchmark longbench --dataset single --samples-per-dataset 8 --num-particles 8 --budget 128 --utility
# python compute_validation.py --config "$REPO_ROOT/config/lorekv-ent-unif-topk.conf" --benchmark ruler     --dataset 4096   --samples-per-dataset 8 --num-particles 8 --budget 128 --utility
# python report_validation.py --data validation_data_<tag>.npz

# --- The budget x selection cross at a FIXED reliability cell. This prices
#     allocation and selection, not the weighting -- the orthogonal sweep to the
#     one above. Set VARIANTS to run it from the ACTIVE block instead:
#       sbatch --export=ALL,VARIANTS="unif-topk adap-topk unif-div adap-div" \\
#              scripts/run_h100_aces.sh ---
# for v in unif-topk adap-topk unif-div adap-div; do cfg="lorekv-ent-$v"; echo "===== Running $cfg ====="; EXPERIMENT=LongBench CONFIG_FILE="$REPO_ROOT/config/$cfg.conf" BUDGET=128 bash "$REPO_ROOT/scripts/run_h100.sh"; done

# --- Score-level figures (S1/S3/S5): re-weight captured particles, no re-decoding ---
# FIGURES="1 10" CONFIG_FILE=config/lorekv-ent-unif-topk.conf BUDGET=128 BENCHMARK=longbench bash "$REPO_ROOT/scripts/run_figures_aces.sh"

# =============================================================================
# LEGACY sweeps (pre-reparametrization). These still run: run_h100.sh maps a config
# that sets LAMBDA_H/PARTICLE_ETA onto (BETA, GAMMA, LAMBDA_D) and logs the mapping.
#   tlckv-pe-*  ->  (1, 1.3, 0)      tlckv-p-*  ->  (1, 1, 0)      NOT a no-penalty cell
#   tlckv-pd-*  ->  (1, 1,   0.3)    i.e. disagreement ON TOP of a full entropy penalty
# Note also that the -mx- configs include M=2, where LAMBDA_D has no effect at all
# (D_1 = D_2 by symmetry, so sd(D) = 0 and the z-score is 0/0; the code returns zeros).
# =============================================================================

# for cfg in config/lorekv-ent-adap-topk.conf config/lorekv-ent-adap-div.conf config/lorekv-ent-adap-topk-llama.conf config/lorekv-ent-adap-div-llama.conf  config/lorekv-dis-adap-topk.conf config/lorekv-dis-adap-topk-llama.conf  config/lorekv-dis-unif-topk.conf config/lorekv-dis-unif-topk-llama.conf; do echo "===== Running $cfg ====="; EXPERIMENT=NIAH CONFIG_FILE="$cfg" BUDGET="128 256 512 1024" bash "scripts/run_h100_aces.sh"; done


# ////////////////// Batch submissions
# EXPERIMENT=LongBench CONFIG_FILE=conf/lorekv-ent-adap-topk BUDGET="128 256 512 1024" bash scripts/run_h100.sh
# EXPERIMENT=RULER     BUDGET=128 sbatch scripts/run_h100_aces.sh
# EXPERIMENT=NIAH      BUDGET=128 sbatch scripts/run_h100_aces.sh

# for cfg in  config/lorekv-dis-unif-topk.conf config/lorekv-dis-adap-topk.conf; do echo "===== Running $cfg ====="; EXPERIMENT=LongBench CONFIG_FILE="$cfg" NUM_PARTICLES="0 1" BUDGET="512 1024" sbatch "scripts/run_h100_aces.sh"; done

# for cfg in  config/lorekv-dis-unif-topk.conf config/lorekv-dis-adap-topk.conf; do echo "===== Running $cfg ====="; EXPERIMENT=LongBench CONFIG_FILE="$cfg" NUM_PARTICLES="2 8" BUDGET="128 256" sbatch "scripts/run_h100_aces.sh"; done

# for cfg in  config/lorekv-dis-unif-topk-llama.conf; do echo "===== Running $cfg ====="; EXPERIMENT=LongBench CONFIG_FILE="$cfg" NUM_PARTICLES="0 1 4 2 8" BUDGET="128 256 512 1024" sbatch "scripts/run_h100_aces.sh"; done


# for cfg in config/lorekv-*.conf; do echo "===== Running $cfg ====="; EXPERIMENT=LongBench CONFIG_FILE="$cfg" NUM_PARTICLES="0 1 2 4 8" BUDGET="128" sbatch "scripts/run_h100_aces.sh"; done


# for cfg in config/lorekv-dis-adap-div-llama.conf config/lorekv-dis-adap-div.conf config/lorekv-dis-adap-topk-llama.conf config/lorekv-dis-adap-topk.conf config/lorekv-dis-unif-div-llama.conf config/lorekv-dis-unif-div.conf config/lorekv-dis-unif-topk-llama.conf config/lorekv-dis-unif-topk.conf config/lorekv-ent-adap-div-llama.conf config/lorekv-ent-adap-div.conf config/lorekv-ent-adap-topk-llama.conf config/lorekv-ent-adap-topk.conf config/lorekv-ent-only-adap-div-llama.conf config/lorekv-ent-only-adap-div.conf config/lorekv-ent-only-adap-topk-llama.conf config/lorekv-ent-only-adap-topk.conf config/lorekv-ent-only-unif-div-llama.conf config/lorekv-ent-only-unif-div.conf config/lorekv-ent-only-unif-topk-llama.conf config/lorekv-ent-only-unif-topk.conf config/lorekv-ent-unif-div-llama.conf config/lorekv-ent-unif-div.conf config/lorekv-ent-unif-topk-llama.conf config/lorekv-ent-unif-topk.conf config/lorekv-logp-adap-div-llama.conf config/lorekv-logp-adap-div.conf config/lorekv-logp-adap-topk-llama.conf config/lorekv-logp-adap-topk.conf config/lorekv-logp-unif-div-llama.conf config/lorekv-logp-unif-div.conf config/lorekv-logp-unif-topk-llama.conf config/lorekv-logp-unif-topk.conf config/lorekv-typ-adap-div-llama.conf config/lorekv-typ-adap-div.conf config/lorekv-typ-adap-topk-llama.conf config/lorekv-typ-adap-topk.conf config/lorekv-typ-dis-adap-div-llama.conf config/lorekv-typ-dis-adap-div.conf config/lorekv-typ-dis-adap-topk-llama.conf config/lorekv-typ-dis-adap-topk.conf config/lorekv-typ-dis-unif-div-llama.conf config/lorekv-typ-dis-unif-div.conf config/lorekv-typ-dis-unif-topk-llama.conf config/lorekv-typ-dis-unif-topk.conf config/lorekv-typ-unif-div-llama.conf config/lorekv-typ-unif-div.conf config/lorekv-typ-unif-topk-llama.conf config/lorekv-typ-unif-topk.conf config/lorekv-uniform-adap-div-llama.conf config/lorekv-uniform-adap-div.conf config/lorekv-uniform-adap-topk-llama.conf config/lorekv-uniform-adap-topk.conf config/lorekv-uniform-unif-div-llama.conf config/lorekv-uniform-unif-div.conf config/lorekv-uniform-unif-topk-llama.conf config/lorekv-uniform-unif-topk.conf; do echo "===== Running $cfg ====="; EXPERIMENT=LongBench CONFIG_FILE="$cfg" NUM_PARTICLES="0 1 2 4 8" BUDGET="128" sbatch "scripts/run_h100_aces.sh"; done


# for cfg in config/lorekv-dis-adap-div-llama.conf config/lorekv-dis-adap-div.conf config/lorekv-dis-adap-topk-llama.conf config/lorekv-dis-adap-topk.conf config/lorekv-dis-unif-div-llama.conf config/lorekv-dis-unif-div.conf config/lorekv-dis-unif-topk-llama.conf config/lorekv-dis-unif-topk.conf config/lorekv-ent-adap-div-llama.conf config/lorekv-ent-adap-div.conf config/lorekv-ent-adap-topk-llama.conf config/lorekv-ent-adap-topk.conf config/lorekv-ent-unif-div-llama.conf config/lorekv-ent-unif-div.conf config/lorekv-ent-unif-topk-llama.conf config/lorekv-ent-unif-topk.conf; do echo "===== Running $cfg ====="; EXPERIMENT=LongBench CONFIG_FILE="$cfg" NUM_PARTICLES="0 1 2 4 8" BUDGET="128" sbatch "scripts/run_h100_aces.sh"; done


# for cfg in config/lorekv-dis-adap-div.conf; do echo "===== Running $cfg ====="; EXPERIMENT=LongBench CONFIG_FILE="$cfg" NUM_PARTICLES="0 1 2 4 8" BUDGET="128" sbatch "scripts/run_h100_aces.sh"; done


# for cfg in config/lorekv-ent-unif-topk.conf; do echo "===== Running $cfg ====="; EXPERIMENT=LongBench CONFIG_FILE="$cfg" LOOKAHEAD_LEN="2 4 8 16 32" NUM_PARTICLES="0 1 2 4 8" BUDGET="128" sbatch "scripts/run_h100_aces.sh"; done

# for len in 2 4 8 16 32; do echo "===== Running LOOKAHEAD_LEN=$len ====="; EXPERIMENT=LongBench CONFIG_FILE="config/lorekv-ent-unif-topk.conf" LOOKAHEAD_LEN="$len" NUM_PARTICLES="0 1 2 4 8" BUDGET="128" sbatch scripts/run_h100_aces.sh; done

# EXPERIMENT=LongBench CONFIG_FILE=lorekv-obj-attn BUDGET="128 256 512 1024" sbatch scripts/run_h100_aces.sh
# EXPERIMENT=LongBench CONFIG_FILE=lorekv-obj-unproj BUDGET="128 256 512 1024" sbatch scripts/run_h100_aces.sh