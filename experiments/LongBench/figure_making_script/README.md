# LORE-KV / TLC-KV analysis figures

Standalone scripts that produce the paper's mechanism figures from the **actual
method code** (`causal_lookahead_snapkv`). Each figure is a two-stage pair:

1. `compute_*.py` — runs the model, computes the quantities, and **saves them to a
   `.npz` in the same folder**. Nothing is plotted here.
2. `plot_*.py` — **reads that `.npz`** and writes the `.png`. No model needed.

This separation means you run the expensive stage once on the GPU, then iterate on
the figure locally.

**Self-describing filenames.** Both artifacts embed the config, model, budget, and
dataset so sweeps never overwrite each other:

```
<data_base>__<conf>__<model>__b<budget>__<dataset>.npz
<fig_base>__<conf>__<model>__b<budget>__<dataset>.png     # same tag
# e.g. alignment_data__tlckv-pe-m4-adap-topk__Mistral-7B-Instruct-v0.2__b128__single.npz
#      fig1_future_query_alignment__tlckv-pe-m4-adap-topk__Mistral-7B-Instruct-v0.2__b128__single.png
```

`plot_*.py` auto-picks the **newest** matching `.npz` in its folder and writes the
`.png` beside it with the same tag — so `compute` then `plot` just works.
Overrides: `compute --out <file>` or `--out-dir <dir>`; `plot --data <file>` or
`--data-dir <dir>` (and `--out <file>`).

> **[FIGURES.md](FIGURES.md)** is the scientific companion to this file: per-figure
> question, formula, and why each one matters for the methodology. This README
> covers how to *run* them; FIGURES.md covers what they *mean*.

> These need a GPU (they load Mistral-7B / Llama-3.1-8B and run real inference).
> Run inside the cluster venv, e.g. `source /scratch/ahsan/venvs/kv44/bin/activate`.
> There is **no GPU on the login node**, so they cannot be smoke-tested there.

## Shared backbone — `common.py`
Everything that touches the model. It reuses the proven pipeline:
`collect_and_lookahead()` is a faithful copy of phases 1–2 of
`lookahead_runner.generate_with_lookahead` that stops *before* `compress()`, so the
figures read exactly what eviction sees: the post-RoPE window queries, the M
lookahead particle queries, the compact full K/V, and the per-particle
(log-prob, entropy, weight). The **true** future queries `Q_future` are obtained by
greedy-decoding the answer against the **full** cache, captured non-invasively with
`PostRoPEQueryCapture` (forward-pre-hooks that recompute post-RoPE q; the forward
pass is unchanged). Scoring reuses each layer's own
`TestTimeLookaheadKVCluster._project_and_score`.

## Datasets
`--dataset single` (default) = single-doc QA (qasper, narrativeqa, multifieldqa_en);
`--dataset full` = all 16 English LongBench tasks; or a comma-list of task names.
`--benchmark ruler --dataset 8192` runs the same figure on RULER instead.

## The four figures

| Folder | What it answers | Data → Figure |
|---|---|---|
| `fig1_future_query_alignment/` | Do confidence-weighted **multiple** trajectories approximate the true decoding queries better than prompt-window / single / uniform / log-prob? (**central premise**) | `alignment_data.npz` → `fig1_future_query_alignment.png` |
| `fig2_score_comparison/` | How much does the importance ordering change from attention → single-lookahead → LORE-KV multi-lookahead? | `score_data.npz` → `fig2_score_comparison.png` |
| `fig3_cache_distribution/` | Does the LORE-KV-evicted cache reproduce the **full-cache** decode-time attention output distribution better than attention-only eviction? | `distribution_data.npz` → `fig3_cache_distribution.png` |
| `fig4_entropy_trajectory/` | Why test-time compute? A single trajectory can land in a high-entropy branch that hurts eviction; entropy-weighting many trajectories fixes it. | `entropy_data.npz` → `fig4_entropy_trajectory.png` |
| `fig5_confidence_calibration/` | Do the log-prob/entropy weights actually pick useful trajectories? Calibration curve, risk–coverage, and Spearman of each signal (log-prob, −entropy, combined `R_m`, weight) vs oracle particle utility. | `calibration_data.npz` → `fig5_confidence_calibration.png` |
| `fig6_mismatch_vs_gain/` | Does LORE-KV help most where prompt and future queries differ most? Per-example scatter of mismatch `1−align(Q_W,Q_true)` vs recall/fidelity gain, with regression + Spearman. | `mismatch_data.npz` → `fig6_mismatch_vs_gain.png` |
| `fig7_retained_token_composition/` | What token classes does each method keep (answer/evidence, entities, recent, sinks…)? Retention rate + enrichment, attention vs AnDPro-window vs LORE-KV. | `composition_data.npz` → `fig7_retained_token_composition.png` |
| `fig8_layer_head_analysis/` | Layer perturbation energy `E_l`, per-head future-query recall gain `G_{l,h}` heatmap, per-head mismatch — where the future-query signal concentrates. | `layer_head_data.npz` → `fig8_layer_head_analysis.png` |
| `fig9_budget_equivalence/` | Continuous budget-equivalence: fidelity `F(B)` per method → equivalent baseline budget and % memory saving at equal full-cache fidelity. | `budget_data.npz` → `fig9_budget_equivalence.png` |

**Terminology note (fig5):** the confidence uses ordinary categorical **next-token
entropy**, not **semantic entropy** (which groups meaning-equivalent generations).
The paper text should keep these distinct — they are not interchangeable.

## One-shot driver (recommended) — `scripts/run_figures.sh`

Runs every figure's `compute_*.py` then `plot_*.py` with the same scratch /
Hugging Face / GPU environment exports as `scripts/run_h100.sh`:

```bash
sbatch scripts/run_figures.sh                        # all 9, LongBench single-doc QA, B=128
bash   scripts/run_figures.sh                        # same, interactive
FIGURES="1 2 6" bash scripts/run_figures.sh          # subset
BENCHMARK=ruler DATASET=8192 bash scripts/run_figures.sh
BUDGET=256 SAMPLES=5 CONFIG_FILE=config/tlckv-pe-mx-unif-topk.conf bash scripts/run_figures.sh
PLOT_ONLY=1 bash scripts/run_figures.sh              # re-draw from saved .npz (no GPU)
```

Knobs (all optional env vars): `BENCHMARK` (`longbench`|`ruler`), `DATASET`,
`BUDGET`, `SAMPLES`, `CONFIG_FILE`, `FIGURES`, `PLOT_ONLY`.

## Running one figure by hand
```bash
cd experiments/LongBench/figure_making_script

# Figure 1 (uses config hyper-parameters: eta, lambda_H, T, ...)
python fig1_future_query_alignment/compute_alignment.py \
    --config ../../../config/tlckv-pe-mx-unif-topk.conf \
    --dataset single --samples-per-dataset 3 --budget 128
python fig1_future_query_alignment/plot_alignment.py

# same pattern for every figure (compute_*.py then plot_*.py)
```

Every compute script takes `--config <repo .conf>` (default
`config/tlckv-pe-m4-adap-topk.conf`), `--budget`, `--samples-per-dataset`,
`--benchmark`, `--dataset`, and `--out`. See each file's docstring for the extras
(e.g. `--m-list`, `--decode-steps`, `--num-particles`).
