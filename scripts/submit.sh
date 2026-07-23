#!/bin/bash
# Usage: bash scripts/submit.sh <config-name>
#   LongBench (default):  bash scripts/submit.sh tlckv-m1-adap-topk
#   RULER:                EXPERIMENT=RULER bash scripts/submit.sh tlckv-m1-adap-topk
set -euo pipefail

NAME="${1:?Usage: $0 <config-name>  (e.g. tlckv-m1-adap-topk)}"
# Which benchmark to run (experiments/<EXPERIMENT>/). Selects the dataset via
# that experiment's run.sh; the config file only carries method hyperparameters.
EXPERIMENT="${EXPERIMENT:-LongBench}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/../config/${NAME}.conf"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: config not found: $CONFIG_FILE" >&2
    exit 1
fi

if [[ ! -d "$SCRIPT_DIR/../experiments/$EXPERIMENT" ]]; then
    echo "Error: experiment not found: experiments/$EXPERIMENT" >&2
    exit 1
fi

JOB_NAME="${EXPERIMENT}-${NAME}"
echo "Submitting: job=$JOB_NAME  experiment=$EXPERIMENT  config=$CONFIG_FILE"
sbatch --job-name="$JOB_NAME" \
    --export=ALL,EXPERIMENT="$EXPERIMENT",CONFIG_FILE="$CONFIG_FILE" \
    "$SCRIPT_DIR/run_h100.sh"
