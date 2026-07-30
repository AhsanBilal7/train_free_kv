#!/bin/bash

# --- Load central path/env configuration (.env) when run standalone ---
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
[[ -f "$_HERE/../../.env" ]] && source "$_HERE/../../.env"
set +a

# ============================================
# Path Configuration (override via env vars, or edit the defaults below)
# ============================================
MODEL_PATH="${MODEL_PATH:-mistralai/Mistral-7B-Instruct-v0.2}"

# Short tag for output folder names, derived from the model (basename of the HF
# id / local path). Keeps each model's pred/ results separate. Override with
# MODEL_TAG=... if you want a custom label.
MODEL_TAG="${MODEL_TAG:-$(basename "$MODEL_PATH")}"
# NIAH builds its haystack from local .txt files (Paul Graham essays), not an HF
# dataset — HAYSTACK_DIR points at that directory (default: ./PaulGrahamEssays).
HAYSTACK_DIR="${HAYSTACK_DIR:-$_HERE/PaulGrahamEssays}"

GPU_ID="${GPU_ID:-4}"
echo "run.sh using GPU_ID=$GPU_ID"

# ============================================
# Sweep arrays — every parameter supports multiple space-separated values.
# Single value = one run; multiple values = one run per combination.
# Set via env vars from the launcher or directly in config/default.conf.
# ============================================

# --- Shared ---
budget_scopes=(${BUDGET:-128})
chunk_size_scopes=(4)
mode_scopes=(${MODE:-andpro})
lambda_red_scopes=(${LAMBDA_RED:-0.0})

# --- causal_lookahead (ignored by other modes) ---
num_particles_scopes=(${NUM_PARTICLES:-4})
lookahead_len_scopes=(${LOOKAHEAD_LEN:-8})
lambda_lookahead_scopes=(${LAMBDA_LOOKAHEAD:-1.0})
lookahead_temperature_scopes=(${LOOKAHEAD_TEMPERATURE:-1.0})
lambda_h_scopes=(${LAMBDA_H:-0.0})
particle_eta_scopes=(${PARTICLE_ETA:-1.0})
robust_alpha_scopes=(${ROBUST_ALPHA:-0.5})
robust_topk_beta_scopes=(${ROBUST_TOPK_BETA:-0.5})
lambda_k_scopes=(${LAMBDA_K:-0.0})
lambda_v_scopes=(${LAMBDA_V:-0.0})
budget_tau_scopes=(${BUDGET_TAU:-1.0})

# --- NIAH sweep dimensions ---
# DATASET_NAME_LIST holds the target CONTEXT LENGTHS in tokens (mirrors RULER's
# "subsets"): one pred/<out_name>/<length>.jsonl is written per length.
dataset_name_list=(${DATASET_NAME_LIST:-"4096" "8192" "16384"})
# NIAH_DEPTHS: needle depth percentages (0..100) swept within each length.
depth_percents=(${NIAH_DEPTHS:-0 10 20 30 40 50 60 70 80 90 100})

# --- The needle / question (override to test a different fact) ---
NEEDLE="${NIAH_NEEDLE:-The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.}"
NEEDLE_ANSWER="${NIAH_ANSWER:-eat a sandwich and sit in Dolores Park}"
QUESTION="${NIAH_QUESTION:-What is the best thing to do in San Francisco?}"
MAX_GEN="${NIAH_MAX_GEN:-64}"

# max_length: hard cap on the fed prompt. Keep it above the largest target length.
MAX_LENGTH="${NIAH_MAX_LENGTH:-64000}"

for budget in "${budget_scopes[@]}"; do
  for chunk_size in "${chunk_size_scopes[@]}"; do
    for mode in "${mode_scopes[@]}"; do
      for lambda_red in "${lambda_red_scopes[@]}"; do

        if [ "$mode" == "causal_lookahead" ]; then

          for num_particles in "${num_particles_scopes[@]}"; do
            for lookahead_len in "${lookahead_len_scopes[@]}"; do
              for lambda_lookahead in "${lambda_lookahead_scopes[@]}"; do
                for lookahead_temperature in "${lookahead_temperature_scopes[@]}"; do
                  for lambda_h in "${lambda_h_scopes[@]}"; do
                    for particle_eta in "${particle_eta_scopes[@]}"; do
                      for robust_alpha in "${robust_alpha_scopes[@]}"; do
                        for robust_topk_beta in "${robust_topk_beta_scopes[@]}"; do
                          for lambda_k in "${lambda_k_scopes[@]}"; do
                            for lambda_v in "${lambda_v_scopes[@]}"; do
                              for budget_tau in "${budget_tau_scopes[@]}"; do

                                echo "budget=$budget chunk=$chunk_size mode=$mode" \
                                     "M=$num_particles T=$lookahead_len lL=$lambda_lookahead" \
                                     "lT=$lookahead_temperature lH=$lambda_h eta=$particle_eta" \
                                     "alpha=$robust_alpha beta=$robust_topk_beta" \
                                     "lK=$lambda_k lV=$lambda_v tau=$budget_tau"

                                out_name="${mode}_${MODEL_TAG}_${budget}_chunk_${chunk_size}_M_${num_particles}_T_${lookahead_len}_lL_${lambda_lookahead}_lT_${lookahead_temperature}_lH_${lambda_h}_eta_${particle_eta}_alpha_${robust_alpha}_beta_${robust_topk_beta}_lK_${lambda_k}_lV_${lambda_v}_tau_${budget_tau}"

                                CUDA_VISIBLE_DEVICES=$GPU_ID python pred.py -m $MODEL_PATH \
                                  --max_length $MAX_LENGTH \
                                  --mode $mode \
                                  --budget $budget \
                                  --chunk_size $chunk_size \
                                  --num_particles $num_particles \
                                  --lookahead_len $lookahead_len \
                                  --lookahead_temperature $lookahead_temperature \
                                  --lambda_h $lambda_h \
                                  --particle_eta $particle_eta \
                                  --robust_alpha $robust_alpha \
                                  --robust_topk_beta $robust_topk_beta \
                                  --lambda_lookahead $lambda_lookahead \
                                  --lambda_k $lambda_k \
                                  --lambda_v $lambda_v \
                                  --budget_tau $budget_tau \
                                  --out_name "$out_name" \
                                  --haystack_dir "$HAYSTACK_DIR" \
                                  --needle "$NEEDLE" \
                                  --needle_answer "$NEEDLE_ANSWER" \
                                  --question "$QUESTION" \
                                  --max_gen $MAX_GEN \
                                  --depth_percents "${depth_percents[@]}" \
                                  --dataset_name_list "${dataset_name_list[@]}"

                              done  # budget_tau
                            done  # lambda_v
                          done  # lambda_k
                        done  # robust_topk_beta
                      done  # robust_alpha
                    done  # particle_eta
                  done  # lambda_h
                done  # lookahead_temperature
              done  # lambda_lookahead
            done  # lookahead_len
          done  # num_particles

        else

          echo "budget=$budget chunk=$chunk_size mode=$mode lambda_red=$lambda_red"

          CUDA_VISIBLE_DEVICES=$GPU_ID python pred.py -m $MODEL_PATH \
            --max_length $MAX_LENGTH \
            --mode $mode \
            --budget $budget \
            --chunk_size $chunk_size \
            --lambda_red $lambda_red \
            --out_name "${mode}_${MODEL_TAG}_${budget}_chunk_${chunk_size}_red_${lambda_red}" \
            --haystack_dir "$HAYSTACK_DIR" \
            --needle "$NEEDLE" \
            --needle_answer "$NEEDLE_ANSWER" \
            --question "$QUESTION" \
            --max_gen $MAX_GEN \
            --depth_percents "${depth_percents[@]}" \
            --dataset_name_list "${dataset_name_list[@]}"

        fi

      done  # lambda_red
    done  # mode
  done  # chunk_size
done  # budget

python eval.py
