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
DATASET_PATH="${DATASET_PATH:-simonjegou/ruler}"

# Short tag for output folder names, derived from the model (basename of the HF
# id / local path). Keeps each model's pred/ results separate. Override with
# MODEL_TAG=... if you want a custom label.
MODEL_TAG="${MODEL_TAG:-$(basename "$MODEL_PATH")}"

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
# Eq. 3 reliability channels: R_m = beta*cbar_m - gamma*Hbar_m - lambda_D*Dhat_m.
#   beta      typicality  (centered log-prob, E[c_u|h_u] = 0)
#   gamma     entropy     (gamma = 1 + LAMBDA_H_old reproduces a published run)
#   lambda_D  disagreement
#   normalize varentropy-normalized typicality (ablated switch, off by default)
# scripts/run_h100.sh maps legacy LAMBDA_H/PARTICLE_ETA onto these if they are unset.
beta_scopes=(${BETA:-1.0})
gamma_scopes=(${GAMMA:-0.0})
lambda_d_scopes=(${LAMBDA_D:-0.0})
normalize_typicality_scopes=(${NORMALIZE_TYPICALITY:-0})
robust_alpha_scopes=(${ROBUST_ALPHA:-0.5})
robust_topk_beta_scopes=(${ROBUST_TOPK_BETA:-0.5})
lambda_k_scopes=(${LAMBDA_K:-0.0})
lambda_v_scopes=(${LAMBDA_V:-0.0})
budget_tau_scopes=(${BUDGET_TAU:-1.0})
# App. C (iii) perturbation objective: what Eq. 14 measures per (query, candidate).
#   projected    ||W_O Delta o||  -- the core method (default; folder name unchanged)
#   unprojected  ||Delta o||      -- identical score with W_O dropped
#   attention    a_{q,i}          -- accumulated attention mass
# Sweepable: SCORE_OBJECTIVE="projected unprojected attention"
score_objective_scopes=(${SCORE_OBJECTIVE:-projected})

# --- RULER "subsets" are the sequence-length configs on the Hub. Each config
#     ("4096" / "8192" / "16384") bundles all 13 RULER tasks (500 rows each). ---
dataset_name_list=(${DATASET_NAME_LIST:-"4096" "8192" "16384"})

for budget in "${budget_scopes[@]}"; do
  for chunk_size in "${chunk_size_scopes[@]}"; do
    for mode in "${mode_scopes[@]}"; do
      for lambda_red in "${lambda_red_scopes[@]}"; do

        if [ "$mode" == "causal_lookahead" ]; then

          for num_particles in "${num_particles_scopes[@]}"; do
            for lookahead_len in "${lookahead_len_scopes[@]}"; do
              for lambda_lookahead in "${lambda_lookahead_scopes[@]}"; do
                for lookahead_temperature in "${lookahead_temperature_scopes[@]}"; do
                  for beta in "${beta_scopes[@]}"; do
                   for gamma in "${gamma_scopes[@]}"; do
                    for lambda_d in "${lambda_d_scopes[@]}"; do
                     for normalize_typicality in "${normalize_typicality_scopes[@]}"; do
                      for robust_alpha in "${robust_alpha_scopes[@]}"; do
                        for robust_topk_beta in "${robust_topk_beta_scopes[@]}"; do
                          for lambda_k in "${lambda_k_scopes[@]}"; do
                            for lambda_v in "${lambda_v_scopes[@]}"; do
                              for budget_tau in "${budget_tau_scopes[@]}"; do
                               for score_objective in "${score_objective_scopes[@]}"; do

                                echo "budget=$budget chunk=$chunk_size mode=$mode" \
                                     "M=$num_particles T=$lookahead_len lL=$lambda_lookahead" \
                                     "lT=$lookahead_temperature beta=$beta gamma=$gamma lD=$lambda_d nrm=$normalize_typicality" \
                                     "alpha=$robust_alpha kappa=$robust_topk_beta" \
                                     "lK=$lambda_k lV=$lambda_v tau=$budget_tau obj=$score_objective"

                                # Canonical output-folder name. scripts/run_naming.py is the
                                # single source of truth -- collect_results.py and eval.py
                                # parse with the same module, so the name can never drift
                                # from what they expect. The hyphenated head is exactly the
                                # run's config-file basename, DERIVED FROM THE NUMBERS
                                # ACTUALLY USED, so a CLI sweep (e.g. LAMBDA_D="0.0 0.3")
                                # writes a different cell per value instead of lying.
                                out_name="$(python ../../scripts/run_naming.py \
                                  --mode "$mode" --model-tag "$MODEL_TAG" \
                                  --budget "$budget" --chunk-size "$chunk_size" \
                                  --num-particles "$num_particles" --lookahead-len "$lookahead_len" \
                                  --lambda-lookahead "$lambda_lookahead" \
                                  --lookahead-temperature "$lookahead_temperature" \
                                  --beta "$beta" --gamma "$gamma" --lambda-d "$lambda_d" \
                                  --normalize-typicality "$normalize_typicality" \
                                  --robust-alpha "$robust_alpha" --robust-topk-beta "$robust_topk_beta" \
                                  --lambda-k "$lambda_k" --lambda-v "$lambda_v" \
                                  --budget-tau "$budget_tau" \
                                  --score-objective "$score_objective")"

                                CUDA_VISIBLE_DEVICES=$GPU_ID python pred.py -m $MODEL_PATH \
                                  --max_length 64000 \
                                  --dataset $DATASET_PATH \
                                  --mode $mode \
                                  --budget $budget \
                                  --chunk_size $chunk_size \
                                  --num_particles $num_particles \
                                  --lookahead_len $lookahead_len \
                                  --lookahead_temperature $lookahead_temperature \
                                  --beta $beta \
                                  --gamma $gamma \
                                  --lambda_d $lambda_d \
                                  --normalize_typicality $normalize_typicality \
                                  --robust_alpha $robust_alpha \
                                  --robust_topk_beta $robust_topk_beta \
                                  --lambda_lookahead $lambda_lookahead \
                                  --lambda_k $lambda_k \
                                  --lambda_v $lambda_v \
                                  --budget_tau $budget_tau \
                                  --score_objective $score_objective \
                                  --out_name "$out_name" \
                                  --dataset_name_list "${dataset_name_list[@]}"

                                # Optional: compute + save analysis figures (npz + png) into this
                                # run's folder, per-length and combined. Opt-in via MAKE_FIGURES=1.
                                if [ "${MAKE_FIGURES:-0}" = "1" ]; then
                                  FIG_BENCHMARK=ruler FIG_BUDGET="$budget" FIG_CHUNK="$chunk_size" \
                                    FIG_RUN_DIR="$(pwd)/pred/${out_name}" FIG_DATASETS="${dataset_name_list[*]}" \
                                    FIG_NUM_PARTICLES="$num_particles" FIG_LOOKAHEAD_LEN="$lookahead_len" \
                                    FIG_LOOKAHEAD_TEMPERATURE="$lookahead_temperature" FIG_BETA="$beta" FIG_GAMMA="$gamma" FIG_LAMBDA_D="$lambda_d" \
                                    FIG_NORMALIZE_TYPICALITY="$normalize_typicality" FIG_ROBUST_ALPHA="$robust_alpha" \
                                    FIG_ROBUST_TOPK_BETA="$robust_topk_beta" FIG_LAMBDA_LOOKAHEAD="$lambda_lookahead" \
                                    FIG_LAMBDA_K="$lambda_k" FIG_LAMBDA_V="$lambda_v" FIG_BUDGET_TAU="$budget_tau" \
                                    FIG_LAMBDA_RED="$lambda_red" FIG_GPU="$GPU_ID" \
                                    bash ../../scripts/make_figures_for_run.sh || true
                                fi

                               done  # score_objective
                              done  # budget_tau
                            done  # lambda_v
                          done  # lambda_k
                        done  # robust_topk_beta
                      done  # robust_alpha
                     done  # normalize_typicality
                    done  # lambda_d
                   done  # gamma
                  done  # beta
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
            --dataset_name_list "${dataset_name_list[@]}"

        fi

      done  # lambda_red
    done  # mode
  done  # chunk_size
done  # budget

python eval.py
