#!/bin/bash

# --- Load central path/env configuration (.env) when run standalone ---
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
[[ -f "$_HERE/../../.env" ]] && source "$_HERE/../../.env"
set +a

# Set PyTorch library path (derived from the active venv, works on any cluster)
# TORCH_LIB_DIR="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
# export LD_LIBRARY_PATH="$TORCH_LIB_DIR:${LD_LIBRARY_PATH:-}"

# ============================================
# Path Configuration (override via env vars, or edit the defaults below)
# ============================================
MODEL_PATH="${MODEL_PATH:-mistralai/Mistral-7B-Instruct-v0.2}"

# Short tag for output folder names, derived from the model (basename of the HF
# id / local path). Keeps each model's pred/ results separate. Override with
# MODEL_TAG=... if you want a custom label.
MODEL_TAG="${MODEL_TAG:-$(basename "$MODEL_PATH")}"
DATASET_PATH="${DATASET_PATH:-THUDM/LongBench}"

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

dataset_name_list=(${DATASET_NAME_LIST:-"qasper" "narrativeqa" "multifieldqa_en" \
    "hotpotqa" "2wikimqa" "musique" \
    "trec" "triviaqa" "samsum" \
    "gov_report" "qmsum" "multi_news" \
    "passage_count" "passage_retrieval_en" \
    "lcc" "repobench-p"})

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
                                  --max_length 64000 \
                                  --dataset $DATASET_PATH \
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
                                  --compress_args_path "c${budget}_w32_k7_maxpool.json" \
                                  --dataset_name_list "${dataset_name_list[@]}"

                                # Optional: compute + save analysis figures (npz + png) into this
                                # run's folder, per-dataset and combined. Opt-in via MAKE_FIGURES=1.
                                if [ "${MAKE_FIGURES:-0}" = "1" ]; then
                                  FIG_BENCHMARK=longbench FIG_BUDGET="$budget" FIG_CHUNK="$chunk_size" \
                                    FIG_RUN_DIR="$(pwd)/pred/${out_name}" FIG_DATASETS="${dataset_name_list[*]}" \
                                    FIG_NUM_PARTICLES="$num_particles" FIG_LOOKAHEAD_LEN="$lookahead_len" \
                                    FIG_LOOKAHEAD_TEMPERATURE="$lookahead_temperature" FIG_LAMBDA_H="$lambda_h" \
                                    FIG_PARTICLE_ETA="$particle_eta" FIG_ROBUST_ALPHA="$robust_alpha" \
                                    FIG_ROBUST_TOPK_BETA="$robust_topk_beta" FIG_LAMBDA_LOOKAHEAD="$lambda_lookahead" \
                                    FIG_LAMBDA_K="$lambda_k" FIG_LAMBDA_V="$lambda_v" FIG_BUDGET_TAU="$budget_tau" \
                                    FIG_LAMBDA_RED="$lambda_red" FIG_GPU="$GPU_ID" \
                                    bash ../../scripts/make_figures_for_run.sh || true
                                fi

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
            --max_length 64000 \
            --dataset $DATASET_PATH \
            --mode $mode \
            --budget $budget \
            --chunk_size $chunk_size \
            --lambda_red $lambda_red \
            --out_name "${mode}_${MODEL_TAG}_${budget}_chunk_${chunk_size}_red_${lambda_red}" \
            --compress_args_path "c${budget}_w32_k7_maxpool.json" \
            --dataset_name_list "${dataset_name_list[@]}"

        fi

      done  # lambda_red
    done  # mode
  done  # chunk_size
done  # budget

python eval.py
