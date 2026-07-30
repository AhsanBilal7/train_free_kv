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
# Submits the equivalent of:
#   EXPERIMENT=RULER MODE=andpro BUDGET="128 256" \
#       CONFIG_FILE=config/andpro.conf bash scripts/run_h100.sh
#
# Usage (from the repo root):
#   sbatch scripts/run_h100_aces.sh
#
# Every default below is overridable at submit time (CLI wins over this file,
# which wins over $CONFIG_FILE):
#   sbatch --export=ALL,BUDGET="512 1024" scripts/run_h100_aces.sh
#   sbatch --export=ALL,EXPERIMENT=LongBench,MODE=causal_lookahead,\
# CONFIG_FILE=config/tlckv-m1-adap-topk.conf scripts/run_h100_aces.sh
#
# ACES GPU partitions (sinfo):
#   gpu        2-00:00:00  h100 x{1,3,4,8} on ac[041,045,049,051,055,096], a30 x2
#   gpu_debug     2:00:00  h100 x2 on ac098, a30 x2  <- quick smoke tests
# =============================================================================
#SBATCH --job-name=aces-ruler-andpro
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/u.ab353540/train_free_kv/logs/out_%x_%j.log
#SBATCH --error=/home/u.ab353540/train_free_kv/logs/err_%x_%j.log
# Your only ACES association (sacctmgr show assoc user=$USER). Pinning it keeps
# sbatch from warning "Job submitted with no account set"; change if you get a
# second allocation.
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
export EXPERIMENT="${EXPERIMENT:-RULER}"
# export MODE="${MODE:-andpro}"
export BUDGET="${BUDGET:-512 1024}"
CONFIG_FILE="${CONFIG_FILE:-config/andpro.conf}"
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
# echo "  MODE        = $MODE"
echo "  BUDGET      = $BUDGET"
echo "  CONFIG_FILE = $CONFIG_FILE"
echo "  SCRATCH     = ${SCRATCH:-<unset>}"
echo "=============================================================="

# Hand off to the shared launcher (its own #SBATCH header is inert under bash).
bash "$REPO_ROOT/scripts/run_h100.sh"




# for cfg in config/andpro.conf config/tlckv-m1-adap-div.conf config/tlckv-m1-adap-topk.conf config/tlckv-m1-unif-topk.conf config/tlckv-p-m4-adap-div.conf config/tlckv-p-m4-adap-topk.conf config/tlckv-p-m4-unif-topk.conf config/tlckv-pe-m4-adap-div.conf config/tlckv-pe-m4-adap-topk.conf config/tlckv-pe-m4-unif-topk.conf config/tlckv-pe-mx-adap-div.conf config/tlckv-pe-mx-unif-topk.conf; do echo "===== Running $cfg ====="; EXPERIMENT=NIAH CONFIG_FILE="$cfg" BUDGET=128  bash scripts/run_h100_aces.sh; done



