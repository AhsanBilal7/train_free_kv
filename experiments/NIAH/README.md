# Needle-In-A-Haystack (NIAH) evaluation for TLC-KV / AnDPro

The conventional single-needle **Needle-In-A-Haystack** test popularized by
[Greg Kamradt](https://github.com/gkamradt/LLMTest_NeedleInAHaystack): a short
fact (the *needle*) is hidden at a controlled **depth** inside a long document
(the *haystack*, built from the Paul Graham essays), the context is grown to a
series of target token **lengths**, and the model is asked one retrieval
question. Retrieval quality is read off a `(length × depth)` grid / heatmap.

This harness runs NIAH through the **exact same KV-eviction machinery** as
[LongBench](../LongBench/) and [RULER](../RULER/)
(`andpro` / `causal_forward_only` / `causal_lookahead`), so every config recipe
in the [top-level README](../../README.md) works here too.

- **Haystack:** local `.txt` files under [`PaulGrahamEssays/`](PaulGrahamEssays/)
  (the 49 essays bundled by the original repo), concatenated (cycling if needed)
  and trimmed to each target length. No HF dataset is downloaded.
- **Needle (default):** *"The best thing to do in San Francisco is eat a
  sandwich and sit in Dolores Park on a sunny day."* — answer substring
  `eat a sandwich and sit in Dolores Park`, question *"What is the best thing to
  do in San Francisco?"* Override with `NIAH_NEEDLE` / `NIAH_ANSWER` /
  `NIAH_QUESTION`.
- **Insertion:** at `depth_percent` of the token stream, snapped back to the
  nearest preceding sentence boundary (period) within 100 tokens — faithful to
  `SingleDepthInserter` in the gkamradt repo.
- **Metric:** gkamradt exact-match (`eval.py`): 1.0 if the answer substring
  appears (case-insensitive) in the response, else 0.0. A graded word-recall is
  also recorded (`NIAH_METRIC=recall` to make it the headline / heatmap metric).

> **Tokenization note.** The gkamradt repo does length math with a generic
> `tiktoken` encoding and lets the provider re-tokenize. Here we use the
> **model's own tokenizer** for every length/insertion computation — matching
> how LongBench/RULER `pred.py` measure `context_length` from the model's
> `input_ids` — so the reported target length reflects what the KV cache
> actually holds. Construction logic is otherwise identical.

## Running

```bash
# From repo root — pick the benchmark with EXPERIMENT (uses config/default.conf)
EXPERIMENT=NIAH bash scripts/run_h100.sh

# Or via SLURM
EXPERIMENT=NIAH CONFIG_FILE=config/tlckv-pe-m4-adap-topk.conf sbatch scripts/run_h100.sh

# Or directly (GPU must be set manually)
cd experiments/NIAH
bash run.sh
```

The two NIAH sweep dimensions are:

- `DATASET_NAME_LIST` — target **context lengths** in tokens (default
  `4096 8192 16384`), mirroring RULER's "subsets". One
  `pred/<out_name>/<length>.jsonl` is written per length.
- `NIAH_DEPTHS` — needle **depth percentages** in `[0,100]` (default
  `0 10 20 30 40 50 60 70 80 90 100`), swept within each length.

```bash
# 4k & 8k contexts, depths at the quartiles, budgets 128 & 256, window-only
EXPERIMENT=NIAH MODE=causal_lookahead LAMBDA_LOOKAHEAD=0.0 \
BUDGET="128 256" DATASET_NAME_LIST="4096 8192" NIAH_DEPTHS="0 25 50 75 100" \
bash scripts/run_h100.sh
```

Other knobs: `NIAH_NEEDLE`, `NIAH_ANSWER`, `NIAH_QUESTION`, `NIAH_MAX_GEN`
(answer tokens, default 64), `NIAH_MAX_LENGTH` (prompt cap, default 64000),
`HAYSTACK_DIR` (point at a different corpus).

## Outputs

- Predictions: `pred/<out_name>/<length>.jsonl` — one row per depth with
  `pred`, gold `answer`, `needle`, `question`, `context_length` (target),
  `input_length` (actual tokens fed), `depth_percent`, `actual_depth_percent`.
- Scores: `pred/<out_name>/result.json` — the full `(length × depth)` grid,
  per-length means, and overall mean for both `exact_match` and `recall`.
- Heatmap: `pred/<out_name>/heatmap.png` — the signature NIAH retrieval heatmap
  (depth × length), colored by the headline metric.

## Evaluate results

`run.sh` calls `eval.py` automatically after prediction. To (re)score existing
predictions:

```bash
cd experiments/NIAH
python eval.py                 # exact-match heatmap + result.json
python eval.py --metric recall # graded word-recall as the headline
```
