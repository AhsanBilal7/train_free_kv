# Needle-In-A-Haystack (NIAH) evaluation for TLC-KV / AnDPro

The **Needle-In-A-Haystack** test popularized by
[Greg Kamradt](https://github.com/gkamradt/needle-in-a-haystack): short facts
(the *needles*) are hidden at controlled **depths** inside a long document (the
*haystack*, built from the Paul Graham essays), the context is grown to a series
of target token **lengths**, and the model is asked one retrieval question.
Retrieval quality is read off a `(length × depth)` grid / heatmap.

Ported against commit
[`021385d`](https://github.com/gkamradt/needle-in-a-haystack/tree/021385d68d3202e37893e9d3cd29011c569abe30);
the inserters and the scorer are verified equivalent to
`needlehaystack/inserters/{single_depth,even_spread}.py` and
`needlehaystack/scorers/exact_match.py` by a randomized differential test.

This harness runs NIAH through the **exact same KV-eviction machinery** as
[LongBench](../LongBench/) and [RULER](../RULER/)
(`andpro` / `causal_forward_only` / `causal_lookahead`), so every config recipe
in the [top-level README](../../README.md) works here too.

- **Haystack:** local `.txt` files under [`PaulGrahamEssays/`](PaulGrahamEssays/)
  (the 49 essays bundled by the original repo, ≈149k Llama-3 tokens),
  concatenated (cycling if needed) and trimmed to each target length. No HF
  dataset is downloaded.
- **Prompt:** `f"{context}\n\n{question}"` with an empty system message, exactly
  as `core/runner.py` builds it, then wrapped in the model's chat template.
- **Metric:** reference `ExactMatchScorer` (`eval.py`) — the **fraction of the
  expected fragments** that appear (case-insensitive, whitespace-trimmed) in the
  response. Single-needle cells carry one fragment, so this is the classic
  1.0/0.0 exact match. A graded word-recall is also recorded
  (`NIAH_METRIC=recall` to make it the headline / heatmap metric).

## Tasks

| `NIAH_TASK` | Inserter | Default needle(s) | Question | Score |
|---|---|---|---|---|
| `single` (default) | `SingleDepthInserter` | *"The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."* → fragment `eat a sandwich and sit in Dolores Park` | *"What is the best thing to do in San Francisco?"* | 1.0 / 0.0 |
| `multi` | `EvenSpreadInserter` | the pizza trio — *Figs* / *Prosciutto* / *Goat cheese* *"…is one of the secret ingredients needed to build the perfect pizza."* → fragments `figs`, `prosciutto`, `goat cheese` | *"What are the secret ingredients needed to build the perfect pizza?"* | fraction of 3 fragments recalled |

- **Single insertion:** at `depth_percent` of the token stream, snapped back to
  the nearest preceding sentence boundary within 100 tokens (`MAX_PERIOD_SNAP`).
  `depth=0` → start, `depth=100` → end.
- **Multi insertion:** needle *i* targets `depth + i·(100−depth)/n`, so
  `depth=25` with 3 needles gives 25% / 50% / 75%. Every index and reported
  depth is computed against the **pre-insertion** length, so earlier needles do
  not inflate later depths.
- Multi-needle runs write to `pred/multi_<out_name>/` so they never collide with
  single-needle runs.

> **Tokenization note.** The reference repo does length math with a generic
> `tiktoken` encoding and lets the provider re-tokenize. Here we use the
> **model's own tokenizer** for every length/insertion computation — matching
> how LongBench/RULER `pred.py` measure `context_length` from the model's
> `input_ids` — so the reported target length reflects what the KV cache
> actually holds. Construction logic is otherwise identical.

> **Length accounting note.** The reference trims the haystack to exactly
> `length` and inserts the needles on top (final = `length` + needle tokens; the
> question is not counted). We budget the other way round — haystack =
> `length − buffer − Σ needle tokens` — so the **total prompt** lands at
> ~`length` and a KV-eviction method sees exactly the swept length.

> ⚠️ **Keep every swept length inside the model's positional range and below
> `max_length`.** Mistral-7B-Instruct-v0.2 tops out at 32768; Llama-3.1-8B at
> 131072. A prompt over `max_length` used to be middle-truncated, which deletes
> the needle for every interior depth while still recording the cell under its
> nominal length. `pred.py` now **aborts** instead (`--on_overflow truncate`
> restores the old behaviour), and warns when a target exceeds
> `max_position_embeddings`.

## Running

```bash
# From repo root — pick the benchmark with EXPERIMENT (uses config/default.conf)
EXPERIMENT=NIAH bash scripts/run_h100.sh

# Or via SLURM
EXPERIMENT=NIAH CONFIG_FILE=config/lorekv-ent-adap-topk.conf sbatch scripts/run_h100.sh

# Or directly (GPU must be set manually)
cd experiments/NIAH
bash run.sh
```

The two NIAH sweep dimensions are:

- `DATASET_NAME_LIST` — target **context lengths** in tokens (default
  `4096 8192 16384 32768`), mirroring RULER's "subsets". One
  `pred/<out_name>/<length>.jsonl` is written per length.
- `NIAH_DEPTHS` — needle **depth percentages** in `[0,100]` (default
  `0 10 20 30 40 50 60 70 80 90 100`), swept within each length.

```bash
# 4k & 8k contexts, depths at the quartiles, budgets 128 & 256, window-only
EXPERIMENT=NIAH MODE=causal_lookahead LAMBDA_LOOKAHEAD=0.0 \
BUDGET="128 256" DATASET_NAME_LIST="4096 8192" NIAH_DEPTHS="0 25 50 75 100" \
bash scripts/run_h100.sh

# multi-needle (pizza trio), scored as the fraction of ingredients recalled
EXPERIMENT=NIAH NIAH_TASK=multi BUDGET="128 512" bash scripts/run_h100.sh
```

Other knobs:

| Env var | Meaning |
|---|---|
| `NIAH_TASK` | `single` (default) or `multi` |
| `NIAH_NEEDLE` / `NIAH_ANSWER` / `NIAH_QUESTION` | single-needle fact, scored fragment, question |
| `NIAH_NEEDLES` / `NIAH_FRAGMENTS` / `NIAH_QUESTION_MULTI` | multi-needle equivalents; `\|`-separated lists |
| `NIAH_GUIDANCE` | extra sentence after the question. Default `""` (reference). Set to `"Don't give information outside the document or repeat your findings."` for the old v1 prompt — note the model often echoes it back as its answer |
| `NIAH_MAX_GEN` | answer tokens (default 64) |
| `NIAH_MAX_LENGTH` | prompt cap; auto-derived as `max(lengths) + 2048` if unset |
| `HAYSTACK_DIR` | point at a different corpus |

## Outputs

- Predictions: `pred/<out_name>/<length>.jsonl` — one row per depth with
  `pred`, `expected_fragments` (what the scorer matches), `answer` (same list,
  RULER-compatible), `needles`, `question`, `retrieval_guidance`, `task`,
  `inserter`, `context_length` (target), `input_length` (actual tokens fed),
  `haystack_token_count`, `depth_percent`, `actual_depth_percents` +
  `insertion_token_indices` (one entry per needle), and `seed`.
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
