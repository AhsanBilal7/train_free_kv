# LORE-KV / TLC-KV analysis figures — RULER

These are **thin wrappers** around the shared logic in
`experiments/LongBench/figure_making_script/`. For the per-figure question,
formula, and rationale see
[`../../LongBench/figure_making_script/FIGURES.md`](../../LongBench/figure_making_script/FIGURES.md). They set RULER defaults
(`--benchmark ruler`, `--dataset 8192`) and save the data + figure **into this
folder**. All flags of the underlying scripts pass straight through.

## One-shot driver (recommended)
The repo-level `scripts/run_figures.sh` builds every RULER figure with the correct
scratch / Hugging Face / GPU environment exports:

```bash
BENCHMARK=ruler DATASET=8192 sbatch scripts/run_figures.sh   # all 9 on RULER 8192
BENCHMARK=ruler DATASET=8192 FIGURES="1 2 6" bash scripts/run_figures.sh
BENCHMARK=ruler DATASET=4096 BUDGET=256 SAMPLES=5 bash scripts/run_figures.sh
BENCHMARK=ruler PLOT_ONLY=1 bash scripts/run_figures.sh      # re-draw from saved .npz
```

## Running one figure by hand
Each figure is the same two-stage pair (compute saves `.npz`, plot reads it):

```bash
cd experiments/RULER/figure_making_script

python fig1_future_query_alignment/compute_alignment.py --dataset 8192 --budget 128
python fig1_future_query_alignment/plot_alignment.py

# fig2_score_comparison / fig3_cache_distribution / fig4_entropy_trajectory
# follow the identical compute_*.py then plot_*.py pattern.
```

RULER "datasets" are sequence-length configs on the Hub (`simonjegou/ruler`):
`4096`, `8192`, `16384`. Use `--dataset full` for all three or `--dataset 4096,8192`
for a subset. Everything else (the metrics, the plots, `common.py`) is shared with
the LongBench folder, so see `../../LongBench/figure_making_script/README.md` for the
full description of what each figure means.

Needs a GPU (cluster venv `kv44`); cannot run on the login node.
