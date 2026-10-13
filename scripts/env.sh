#!/bin/bash
# Shared environment for every launcher in scripts/ (sourced, not executed).
#
#   1. Sources $REPO_ROOT/.env when present -- machine-specific paths only
#      (virtual environment, Hugging Face caches, scratch directories); see
#      .env.example. Experiment settings belong in config/*.conf.
#   2. Activates $VENV_PATH when it is set.
#   3. Exposes the GPU(s) SLURM allocated as CUDA_VISIBLE_DEVICES.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# sbatch runs a copy of the script from a spool directory; fall back to the
# directory the job was submitted from.
if [[ ! -d "$REPO_ROOT/experiments" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
fi
export REPO_ROOT

# A .env written for an older version of this repository may export MODEL_PATH or
# DATASET_PATH. Those are experiment settings, and since the environment takes
# precedence over a config file they would silently replace the config's model.
# Keep them only if the caller set them explicitly.
_caller_model="${MODEL_PATH-__unset__}"
_caller_dataset="${DATASET_PATH-__unset__}"
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
fi
[[ "$_caller_model" == "__unset__" ]] && unset MODEL_PATH
[[ "$_caller_dataset" == "__unset__" ]] && unset DATASET_PATH

if [[ -n "${VENV_PATH:-}" ]]; then
    # shellcheck disable=SC1091
    source "$VENV_PATH/bin/activate"
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -n "${SLURM_JOB_GPUS:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$SLURM_JOB_GPUS"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Fewer allocator fragmentation failures on long, variable-length prompts.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$REPO_ROOT/logs"
