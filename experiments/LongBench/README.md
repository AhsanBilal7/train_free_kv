# LongBench evaluation for TLC-KV / AnDPro

Based on [SnapKV](https://github.com/FasterDecoding/SnapKV).
See the [top-level README](../../README.md) for the full TLC-KV configuration reference and named config combinations.

## Running

```bash
# From repo root — uses config/default.conf
bash scripts/run_h100.sh

# Or directly (GPU must be set manually)
cd experiments/LongBench
bash run.sh
```

## Quick config recipes for this benchmark

All commands below run from the **repo root**. Adjust `BUDGET` and `DATASET_NAME_LIST` as needed.

### Prompt-window baseline (no lookahead, no overhead)

```bash
MODE=causal_lookahead LAMBDA_LOOKAHEAD=0.0 \
BUDGET="128 256 512 1024" bash scripts/run_h100.sh
```

### 1a — PE-Adap-Mean (entropy-guided, adaptive budget, mean aggregation)

```bash
MODE=causal_lookahead \
NUM_PARTICLES=4 LOOKAHEAD_LEN=8 LAMBDA_LOOKAHEAD=1.0 LOOKAHEAD_TEMPERATURE=1.0 \
LAMBDA_H=0.3 PARTICLE_ETA=1.0 \
ROBUST_ALPHA=0.0 ROBUST_TOPK_BETA=1.0 \
LAMBDA_K=0.0 LAMBDA_V=0.0 BUDGET_TAU=1.0 \
BUDGET="128 256 512 1024" bash scripts/run_h100.sh
```

### 1b — P-Adap-Mean (log-prob only, adaptive budget, mean aggregation)

```bash
MODE=causal_lookahead \
NUM_PARTICLES=4 LOOKAHEAD_LEN=8 LAMBDA_LOOKAHEAD=1.0 LOOKAHEAD_TEMPERATURE=1.0 \
LAMBDA_H=0.0 PARTICLE_ETA=1.0 \
ROBUST_ALPHA=0.0 ROBUST_TOPK_BETA=1.0 \
LAMBDA_K=0.0 LAMBDA_V=0.0 BUDGET_TAU=1.0 \
BUDGET="128 256 512 1024" bash scripts/run_h100.sh
```

### 1c — S1-Adap-Mean (single particle, adaptive budget, mean aggregation)

```bash
MODE=causal_lookahead \
NUM_PARTICLES=1 LOOKAHEAD_LEN=8 LAMBDA_LOOKAHEAD=1.0 LOOKAHEAD_TEMPERATURE=1.0 \
LAMBDA_H=0.0 PARTICLE_ETA=1.0 \
ROBUST_ALPHA=0.0 ROBUST_TOPK_BETA=1.0 \
LAMBDA_K=0.0 LAMBDA_V=0.0 BUDGET_TAU=1.0 \
BUDGET="128 256 512 1024" bash scripts/run_h100.sh
```

### 2a/2b/2c — Uniform budget variants (set BUDGET_TAU=0.0 on any of the above)

```bash
# Example: P-Unif-Mean
MODE=causal_lookahead \
NUM_PARTICLES=4 LOOKAHEAD_LEN=8 LAMBDA_LOOKAHEAD=1.0 LOOKAHEAD_TEMPERATURE=1.0 \
LAMBDA_H=0.0 PARTICLE_ETA=1.0 \
ROBUST_ALPHA=0.0 ROBUST_TOPK_BETA=1.0 \
LAMBDA_K=0.0 LAMBDA_V=0.0 BUDGET_TAU=0.0 \
BUDGET="128 256 512 1024" bash scripts/run_h100.sh
```

### 3a/3b/3c — Diversity-aware variants (add LAMBDA_K=0.1 LAMBDA_V=0.1 to any of group 1/2)

```bash
# Example: PE-Adap-Mean-Div
MODE=causal_lookahead \
NUM_PARTICLES=4 LOOKAHEAD_LEN=8 LAMBDA_LOOKAHEAD=1.0 LOOKAHEAD_TEMPERATURE=1.0 \
LAMBDA_H=0.3 PARTICLE_ETA=1.0 \
ROBUST_ALPHA=0.0 ROBUST_TOPK_BETA=1.0 \
LAMBDA_K=0.1 LAMBDA_V=0.1 BUDGET_TAU=1.0 \
BUDGET="128 256 512 1024" bash scripts/run_h100.sh
```

## Evaluate results

After `run.sh` finishes (or `pred.py` completes), scores are computed with:

```bash
cd experiments/LongBench
python eval.py
```
