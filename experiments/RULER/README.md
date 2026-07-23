# RULER evaluation for TLC-KV / AnDPro

[RULER](https://github.com/NVIDIA/RULER) generates synthetic examples to evaluate
long-context LLMs at configurable sequence lengths across 13 tasks in 4 categories
(needle-in-a-haystack, multi-hop tracing, aggregation, question answering).

This harness runs RULER through the **exact same KV-eviction machinery** as
[LongBench](../LongBench/) (`andpro` / `causal_forward_only` / `causal_lookahead`),
so every config recipe in the [top-level README](../../README.md) works here too.

- **Dataset:** [`simonjegou/ruler`](https://huggingface.co/datasets/simonjegou/ruler)
  (Hub). Its config names are **sequence lengths** — `4096`, `8192`, `16384` — and
  each config bundles all 13 tasks (500 rows each = 6500 rows).
- **Metric:** RULER string-match (`calculate_metrics.py`): partial match for the
  `qa_*` tasks, all-match for every other task.

## Running

```bash
# From repo root — pick the benchmark with EXPERIMENT (uses config/default.conf)
EXPERIMENT=RULER bash scripts/run_h100.sh

# Or via SLURM
EXPERIMENT=RULER CONFIG_FILE=config/tlckv-pe-m4-adap-topk.conf sbatch scripts/run_h100.sh

# Or directly (GPU must be set manually)
cd experiments/RULER
bash run.sh
```

The only knob that differs from LongBench is the dataset list: for RULER,
`DATASET_NAME_LIST` holds **sequence-length configs** (default `4096 8192 16384`)
instead of task names.

```bash
# Only the 4k and 8k configs, budgets 128 & 256, window-only (no lookahead)
EXPERIMENT=RULER MODE=causal_lookahead LAMBDA_LOOKAHEAD=0.0 \
BUDGET="128 256" DATASET_NAME_LIST="4096 8192" bash scripts/run_h100.sh
```

`DATASET_PATH` (`simonjegou/ruler`) is defined in [run.sh](run.sh) and only needs
overriding to point at a local copy of the dataset.

## Outputs

- Predictions: `pred/<out_name>/<length>.jsonl` — one row per example with
  `pred`, gold `answer`, `task`, `length`.
- Scores: `pred/<out_name>/result.json` — per-task scores, the mean over the 13
  tasks at each length, and the overall mean across lengths.

## Evaluate results

`run.sh` calls `eval.py` automatically after prediction. To (re)score existing
predictions:

```bash
cd experiments/RULER
python eval.py
```
