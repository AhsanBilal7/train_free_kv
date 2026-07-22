#!/bin/bash
# Usage: bash scripts/submit.sh tlckv-m1-adap-topk
set -euo pipefail

NAME="${1:?Usage: $0 <config-name>  (e.g. tlckv-m1-adap-topk)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/../config/${NAME}.conf"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: config not found: $CONFIG_FILE" >&2
    exit 1
fi

echo "Submitting: job=$NAME  config=$CONFIG_FILE"
sbatch --job-name="$NAME" --export=ALL,CONFIG_FILE="$CONFIG_FILE" "$SCRIPT_DIR/run_h100.sh"
