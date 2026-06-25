#!/bin/bash

# Set PyTorch library path (derived from the active venv, works on any cluster)
TORCH_LIB_DIR="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LD_LIBRARY_PATH="$TORCH_LIB_DIR:${LD_LIBRARY_PATH:-}"

# ============================================
# Path Configuration (override via env vars, or edit the defaults below)
# ============================================
# Model path - Update to your model location, or use a Hugging Face Hub id
MODEL_PATH="${MODEL_PATH:-mistralai/Mistral-7B-Instruct-v0.2}"

# Dataset path - Update to your dataset location, or use a Hugging Face Hub id
DATASET_PATH="${DATASET_PATH:-THUDM/LongBench}"

# GPU device ID
GPU_ID="${GPU_ID:-4}"
echo "run.sh using GPU_ID=$GPU_ID"

# ============================================
# Experiment Configuration
# ============================================
budget_scopes=(128 256 512 1024)
chunk_size_scopes=(4)
Mode=(andpro)

# Full English-only LongBench dataset set (matches pred.py's default list,
# minus the Chinese-only sets dureader/vcsum/lsht/passage_retrieval_zh since
# Mistral-7B-Instruct isn't tuned for Chinese).
dataset_name_list=(
    "qasper" "narrativeqa" "multifieldqa_en"
    "hotpotqa" "2wikimqa" "musique"
    "trec" "triviaqa" "samsum"
    "gov_report" "qmsum" "multi_news"
    "passage_count" "passage_retrieval_en"
    "lcc" "repobench-p"
)

for budget in ${budget_scopes[@]}; do
    for chunk_size in ${chunk_size_scopes[@]}; do
        for mode in ${Mode[@]}; do
            echo "budget: "$budget", chunk_size: "$chunk_size", mode: "$mode
        
            CUDA_VISIBLE_DEVICES=$GPU_ID python pred.py -m $MODEL_PATH \
                --max_length 64000 \
                --dataset $DATASET_PATH \
                --mode  $mode \
                --budget $budget \
                --chunk_size $chunk_size \
                --out_name "$mode"_Mistral-7B_"$budget"_chunk_"$chunk_size" \
                --compress_args_path c"$budget"_w32_k7_maxpool.json \
                --dataset_name_list "${dataset_name_list[@]}"
        done
    done
done

python eval.py