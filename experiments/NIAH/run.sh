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

# --- NIAH sweep dimensions ---
# DATASET_NAME_LIST holds the target CONTEXT LENGTHS in tokens (mirrors RULER's
# "subsets"): one pred/<out_name>/<length>.jsonl is written per length.
#
# NOTE ON LENGTHS. Keep every entry inside the model's positional range, or the
# row stops measuring KV eviction and starts measuring RoPE extrapolation:
#   Mistral-7B-Instruct-v0.2  max_position_embeddings = 32768
#   Llama-3.1-8B-Instruct     max_position_embeddings = 131072
# The previous default here was "32768 65536 131072", which is out of range for
# Mistral AND above the 64000 prompt cap below -- every 65536/131072 cell was
# silently middle-truncated to 64000 tokens, deleting the needle for all
# interior depths. pred.py now aborts on that instead of scoring nonsense.
dataset_name_list=(${DATASET_NAME_LIST:-"4096" "8192" "16384" "32768"})

# NIAH_DEPTHS: needle depth percentages (0..100) swept within each length.
depth_percents=(${NIAH_DEPTHS:-0 10 20 30 40 50 60 70 80 90 100})

# --- Task: "single" (one needle at one depth) or "multi" (N needles spread
# evenly from the depth to 100%, scored as the fraction of fragments recalled).
NIAH_TASK="${NIAH_TASK:-single}"

# --- The needle / question (override to test a different fact) ---
NEEDLE="${NIAH_NEEDLE:-The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.}"
NEEDLE_ANSWER="${NIAH_ANSWER:-eat a sandwich and sit in Dolores Park}"
QUESTION="${NIAH_QUESTION:-What is the best thing to do in San Francisco?}"

# --- Multi-needle defaults (the reference "pizza trio"). Each entry of
# NIAH_NEEDLES is one needle; NIAH_FRAGMENTS holds the matching scored
# substrings. Use `|` as the separator so the sentences can contain spaces.
NIAH_NEEDLES="${NIAH_NEEDLES:-Figs are one of the secret ingredients needed to build the perfect pizza.|Prosciutto is one of the secret ingredients needed to build the perfect pizza.|Goat cheese is one of the secret ingredients needed to build the perfect pizza.}"
NIAH_FRAGMENTS="${NIAH_FRAGMENTS:-figs|prosciutto|goat cheese}"
NIAH_QUESTION_MULTI="${NIAH_QUESTION_MULTI:-What are the secret ingredients needed to build the perfect pizza?}"

MAX_GEN="${NIAH_MAX_GEN:-64}"

# retrieval_guidance: appended after the question. The reference runner sends
# just "<context>\n\n<question>", so the default is empty. Set
# NIAH_GUIDANCE="Don't give information outside the document or repeat your findings."
# to restore the old (v1) prompt -- but note the model often echoes that
# sentence back as its answer, which scores as a spurious retrieval failure.
GUIDANCE="${NIAH_GUIDANCE:-}"

# max_length: hard cap on the fed prompt. MUST stay above the largest target
# length or the needle gets truncated away; derived automatically unless the
# caller pins NIAH_MAX_LENGTH.
if [[ -n "${NIAH_MAX_LENGTH:-}" ]]; then
  MAX_LENGTH="$NIAH_MAX_LENGTH"
else
  _max_target=0
  for _L in "${dataset_name_list[@]}"; do
    (( _L > _max_target )) && _max_target=$_L
  done
  MAX_LENGTH=$(( _max_target + 2048 ))
fi
echo "run.sh NIAH task=$NIAH_TASK lengths=(${dataset_name_list[*]}) max_length=$MAX_LENGTH"

# Split the `|`-separated multi-needle strings into bash arrays.
IFS='|' read -r -a needles_arr    <<< "$NIAH_NEEDLES"
IFS='|' read -r -a fragments_arr  <<< "$NIAH_FRAGMENTS"

# Per-task CLI fragment, expanded unquoted-as-array at each call site below.
if [ "$NIAH_TASK" == "multi" ]; then
  task_args=(--task multi
             --needles "${needles_arr[@]}"
             --needle_fragments "${fragments_arr[@]}"
             --question "$NIAH_QUESTION_MULTI")
else
  task_args=(--task single
             --needle "$NEEDLE"
             --needle_answer "$NEEDLE_ANSWER"
             --question "$QUESTION")
fi

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
                                [ "$NIAH_TASK" == "multi" ] && out_name="multi_${out_name}"

                                CUDA_VISIBLE_DEVICES=$GPU_ID python pred.py -m $MODEL_PATH \
                                  --max_length $MAX_LENGTH \
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
                                  --haystack_dir "$HAYSTACK_DIR" \
                                  "${task_args[@]}" \
                                  --retrieval_guidance "$GUIDANCE" \
                                  --max_gen $MAX_GEN \
                                  --depth_percents "${depth_percents[@]}" \
                                  --dataset_name_list "${dataset_name_list[@]}"

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

          out_name="${mode}_${MODEL_TAG}_${budget}_chunk_${chunk_size}_red_${lambda_red}"
          [ "$NIAH_TASK" == "multi" ] && out_name="multi_${out_name}"

          CUDA_VISIBLE_DEVICES=$GPU_ID python pred.py -m $MODEL_PATH \
            --max_length $MAX_LENGTH \
            --mode $mode \
            --budget $budget \
            --chunk_size $chunk_size \
            --lambda_red $lambda_red \
            --out_name "$out_name" \
            --haystack_dir "$HAYSTACK_DIR" \
            "${task_args[@]}" \
            --retrieval_guidance "$GUIDANCE" \
            --max_gen $MAX_GEN \
            --depth_percents "${depth_percents[@]}" \
            --dataset_name_list "${dataset_name_list[@]}"

        fi

      done  # lambda_red
    done  # mode
  done  # chunk_size
done  # budget

python eval.py
