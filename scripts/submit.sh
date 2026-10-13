#!/bin/bash
# Submit one config as a SLURM job.
#
#   bash scripts/submit.sh lorekv-ent-unif-topk                     # LongBench
#   EXPERIMENT=RULER bash scripts/submit.sh lorekv-ent-unif-topk     # RULER
#   BUDGET=128 bash scripts/submit.sh andpro-llama                  # override a key
#
# Extra sbatch options (partition, account, GPU type) go in SBATCH_ARGS, e.g.
#   SBATCH_ARGS="--partition=gpu --gres=gpu:h100:1" bash scripts/submit.sh ...
set -euo pipefail

NAME="${1:?usage: $0 <config-name>  (a file in config/, without .conf)}"
EXPERIMENT="${EXPERIMENT:-LongBench}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="$REPO_ROOT/config/${NAME}.conf"

[[ -f "$CONFIG_FILE" ]] || { echo "config not found: $CONFIG_FILE" >&2; exit 1; }
[[ -d "$REPO_ROOT/experiments/$EXPERIMENT" ]] || { echo "unknown experiment: $EXPERIMENT" >&2; exit 1; }

mkdir -p "$REPO_ROOT/logs"
cd "$REPO_ROOT"
# shellcheck disable=SC2086
sbatch --job-name="${EXPERIMENT}-${NAME}" ${SBATCH_ARGS:-} \
    --export=ALL,EXPERIMENT="$EXPERIMENT",CONFIG_FILE="$CONFIG_FILE" \
    scripts/run_experiment.sh
