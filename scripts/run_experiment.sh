#!/bin/bash
#SBATCH --job-name=lorekv
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Run one config on one benchmark: every combination of the config's sweep
# values is decoded with experiments/<EXPERIMENT>/pred.py and then scored.
#
#   CONFIG_FILE=config/lorekv-ent-unif-topk.conf bash scripts/run_experiment.sh
#   EXPERIMENT=RULER CONFIG_FILE=config/andpro.conf sbatch scripts/run_experiment.sh
#   BUDGET="128 256" NUM_PARTICLES="1 4" CONFIG_FILE=... bash scripts/run_experiment.sh
#
# EXPERIMENT is LongBench (default), RULER or NIAH. Any config key can be
# overridden from the environment; see scripts/sweep.py.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

EXPERIMENT="${EXPERIMENT:-LongBench}"
CONFIG_FILE="${CONFIG_FILE:-$REPO_ROOT/config/lorekv-ent-unif-topk.conf}"
echo "experiment=$EXPERIMENT config=$CONFIG_FILE gpu=$CUDA_VISIBLE_DEVICES"
python "$REPO_ROOT/scripts/sweep.py" --benchmark "$EXPERIMENT" --config "$CONFIG_FILE" "$@"
