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
EXPERIMENT=RULER CONFIG_FILE=config/lorekv-ent-adap-topk.conf sbatch scripts/run_h100.sh

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

## Long context: 64K and 128K

`simonjegou/ruler` ships only `4096`, `8192` and `16384`. The `65536` and
`131072` configs are generated locally by [`long_context/`](long_context/):

| file | role |
|---|---|
| `make_data.sh setup` | clone NVIDIA RULER at `3b8928f` (HEAD when the Hub export was made, 2024-09-30), build a small venv, fetch the corpora (login node: needs network) |
| `make_data.sh generate 65536 131072` | run RULER's `prepare.py` for all 13 tasks x 500 rows, convert, check lengths (CPU job) |
| `make_data.sh validate` | regenerate 5 rows/task at 4096 and require them **byte-identical** to the Hub rows |
| `convert_ruler.py` | RULER `{input, outputs}` jsonl -> the Hub schema (`context`, `question`, `answer_prefix`, `answer`, `task`, `max_new_tokens`) |
| `check_lengths.py` | real prompt length after the chat template, to size `MAX_LENGTH` |
| `progress.py` | rows done + interim scores of in-flight runs |

The pipeline reproduces the Hub data exactly (all 13 tasks, at 4096 and 16384),
so 64K/128K follow the same protocol as the published 4K-16K numbers: same
generator revision, seed 42, Llama-3.1 tokenizer, `base` template, 500 rows per
task. Two things were needed to get there and are pinned in `make_data.sh`:
`wonderwords==2.2.0` (3.x changed the word lists, altering every NIAH key and CWE
word) and the LFS blob of `english_words.json` (a plain clone holds only the
pointer).

Differences from the Hub configs, all deliberate:

- The data is read from `RULER_LOCAL_DIR` (default `$SCRATCH/ruler_long`) for
  any length not on the Hub; `pred.py --local_dataset` / `load_ruler_config`.
- Rows are stored **round-robin across tasks**, so any prefix of an in-flight
  run is task-balanced and `progress.py` gives an unbiased preview. Scoring
  groups by task, so the final number is order-independent.
- `MAX_LENGTH` (default now 131072, was a hardcoded 64000) must exceed the prompt:
  64K prompts are 59.8K-65.6K tokens, so the old cap would have middle-truncated
  almost every one, needles included. Nothing at 4K-16K changes.

```bash
EXPERIMENT=RULER CONFIG_FILE=lorekv-ent-unif-topk-llama BUDGET=128 \
  DATASET_NAME_LIST="65536" MAX_LENGTH=131072 sbatch scripts/run_h100_aces.sh
```

Needs a backbone whose window covers the length: Llama-3.1-8B (131072), not
Mistral-7B-v0.2 (32768). `result.json`'s top-level `average` stays the mean over
4K/8K/16K (the paper's Avg. column); 64K and 128K are reported per length.

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
