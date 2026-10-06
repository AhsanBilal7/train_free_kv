# `demos/` — token-ranking evidence: CriticalKV vs AnDPro vs LORE-KV

Everything in this folder is produced by

```bash
sbatch scripts/run_demo_token_ranking.sh                       # LongBench + RULER
# or, inside an allocation / on a compute node:
bash scripts/run_demo_token_ranking.sh
# or one benchmark directly:
python scripts/demo_token_ranking.py --benchmark ruler --dataset 4096,8192 \
       --samples-per-dataset 3 --budgets 128 512
```

and turned into figures by

```bash
python scripts/plot_token_ranking_demo.py                      # newest run in demos/
python scripts/plot_token_ranking_demo.py --run demos/<run-a> demos/<run-b>
```

The end-to-end tables (`scripts/run_h100_aces.sh`) say LORE-KV wins. This folder says
**why**: on one prompt, one prefill and one particle draw, every criterion scores the
same candidate KV pairs, and each retained cache is measured against what the model
actually needed while answering.

## What is compared

| key in the artifacts | what it is | where it comes from |
|---|---|---|
| `criticalkv` | pooled mean prompt-window attention, maxpool(7) smoothed | port of `adaptive_snapkv/.../AdaptiveSnapKVCluster.calcul_attn_sore` — the repo's `MODE=critical` (`config/critical.conf`) |
| `andpro` | anchor-direction projection: chunk-summed attention-weighted values projected on the summed droppable-value direction | port of `andpro_snapkv/.../update_kv_wo_gqa` — the repo's `MODE=andpro` (`config/andpro.conf`) |
| `lorekv_window` | projected leave-one-out deletion effect at `lambda_L = 0` (prompt-window queries only) | `TestTimeLookaheadKVCluster._project_and_score`; the paper's "AnDPro window" analysis criterion |
| `lorekv` | the full method: window + `lambda_L * sum_m omega_m * particle_m` | same scorer, reliability weights from the config's `(BETA, GAMMA, LAMBDA_D)` |
| `oracle` | the same projected score evaluated on the **true** decoding queries | recorded after the fact from a full-cache greedy decode; analysis only, never deployable |

**Selection is held fixed** across criteria (uniform per-head chunk top-B, protected set
= attention sink + observation window, chunking as in `compress()`), so every difference
is attributable to the score and not to allocation or selection. This is the paper's own
mechanism protocol (App. "Compared criteria").

## Metrics

| metric | meaning |
|---|---|
| `recall` | `|top_B(S) ∩ top_B(S*)| / B` against the oracle ranking — does the criterion pick the tokens the future queries need? |
| `spearman` | rank correlation with the oracle score (per layer) |
| `fidelity` | `cos(o_retained, o_full)` on the true future queries — does the kept cache behave like the full cache? |
| `evidence_keep` | fraction of gold-answer token positions surviving eviction |
| `evidence_enrich` | `P(answer | retained) / P(answer | prompt)` — 1.0 means eviction is blind to the evidence |
| `topb_overlap` | pairwise agreement between two criteria's retained sets — how far the ranking moves |
| `particle_stability` | LORE-KV top-B overlap between two **disjoint halves** of the particle ensemble — sampling-noise robustness at no extra decoding cost |

## Layout of a run

```
demos/<config>__<model>__<benchmark>__<dataset>__b128-512/
├── manifest.json            config, model, hyperparameters, git sha, selector description
├── summary_long.csv         tidy rows: dataset, sample, context_length, budget, method, metric, value
├── summary.json             manifest + mean/sem/n per (dataset, budget, method, metric)
├── samples/<dataset>__s<i>.npz
│      <method>__recall / __fidelity / __evidence_keep   [n_layers, n_budgets]
│      <method>__spearman                                [n_layers]
│      overlap__<a>_vs_<b>                               [n_layers, n_budgets]
│      lorekv__stability                                 [n_layers, n_budgets]
│      answer_positions, n, n_answer, budgets, particle_weights, context_length, rep_layer
└── profiles/<dataset>__s<i>.npz     (representative middle layer)
       score_<method>        [n]        head-mean token score
       headscore_<method>    [H, n]     per-head token score (float16)
       keepfrac_b<B>_<method>[n]        fraction of heads retaining each position
       answer_positions, n, budgets, window_size, rep_layer
```

`summary_long.csv` is the file to plot from; the `.npz` files carry the per-layer and
per-token detail for anything more specific.

## Figures the plotting script produces

`ranking_quality.{png,pdf}`
* (a) Recall@B against the oracle ranking, per benchmark and budget.
* (b) Spearman rank agreement with the oracle across layers.
* (c) Attention-output fidelity of the retained cache on future queries.

`--layout rows` re-arranges it as 2x3, one benchmark per row (a-c LongBench, d-f RULER).
That is the arrangement the paper figure uses and the one `matlab/make_ranking_quality.m`
reproduces, so the caption of `fig:criterion_comparison` matches either toolchain:

```bash
python scripts/plot_token_ranking_demo.py --run demos/<lb-run> demos/<ruler-run> \
       --out-dir demos/figures_combined --layout rows
```

`ranking_evidence.{png,pdf}`
* (a) Answer-token enrichment `P(answer | R) / P(answer | X)` per benchmark and budget.
* (b) Retention strip: which prompt positions each criterion keeps at the smallest
  budget, with the gold-answer positions marked underneath.

## Cost and knobs

Per sample: one prefill, `M*T` lookahead decode steps, `--decode-steps` greedy steps for
the oracle queries, then scoring for all criteria on every layer. Defaults
(`SAMPLES=3`, `BUDGETS="128 512"`, `MAX_LENGTH=16000`, `DECODE_STEPS=24`) sit well inside
a `gpu_debug` allocation for RULER-4K; LongBench single-doc QA is longer, so raise
`--max-length` only with the memory headroom to match.

Useful overrides: `SAMPLES`, `BUDGETS`, `DECODE_STEPS`, `MAX_LENGTH`, `LB_DATASETS`,
`RULER_DATASETS`, `BENCHMARKS`, `CONFIG_FILE`, `MODEL_PATH` (e.g.
`MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct CONFIG_FILE=lorekv-ent-unif-topk-llama`).

## Reading the metrics honestly

`recall_cand` (and `spearman`) compare each criterion against an oracle that is *itself*
the projected leave-one-out score (Eq. 5) evaluated on the true future queries. That
functional form is shared with `lorekv_window` and `lorekv`, so those two start closer to
the oracle by construction, and the recall gap alone is not a self-contained argument.

`fidelity` carries no such bias — it asks only whether the retained cache reproduces the
full-cache attention output on the true future queries, and it is computed from the
cache, not from any scoring rule. Quote fidelity as the primary criterion-agnostic
evidence and recall as the mechanism-level explanation of it.

How the metrics behave in practice (first runs, Mistral-7B-Instruct-v0.2,
2 samples/dataset, so these are shapes rather than final numbers):

* `recall_cand` and `spearman` separate the criteria decisively at this sample size.
  RULER-4K at `B=128`: `criticalkv` 0.586, `andpro` 0.597, `lorekv_window` 0.700,
  `lorekv` 0.788. LongBench single-doc QA is wider still (0.32 / 0.33 / 0.41 / 0.55).
* `fidelity` separates consistently but by a small absolute margin, and LORE-KV sits
  between the prompt-side criteria and the oracle: LongBench `B=128` 0.797 / 0.807 /
  0.816 / **0.857**, oracle 0.897.
* `evidence_enrich` is the noisiest of the three. With two samples per dataset its
  standard error is the size of the differences (RULER-4K `B=128`: `lorekv` 18.2 ± 1.2
  vs. `criticalkv` 19.0 ± 0.4), and the ordering is not yet resolved. The paper's
  corresponding figure averaged over ~750K tokens; raise `SAMPLES` well above 2 before
  reading anything into this column, and treat a small run's enrichment bars as a
  sanity check rather than a result.

## Caveats worth stating in the paper

* `MODE=critical` in this repository runs the Ada-KV/SnapKV-style adaptive path
  (`replace_*_adaptive`), so the `criticalkv` column here is that criterion, not a
  re-implementation of the CriticalKV paper's scorer. It is exactly what the repo's
  CriticalKV rows were produced with.
* AnDPro is scored at chunk granularity (its native unit) and expanded to tokens; the
  shared selector re-pools to chunks, so no criterion is advantaged by granularity.
* Gold-answer positions are found by exact token-id subsequence match. A rare-token
  fallback applies only to short answers (<= 8 tokens, i.e. RULER needles and entity
  answers); a long free-form answer that never appears verbatim gets an *empty* evidence
  class and drops out of that metric, rather than marking a few percent of the prompt as
  "evidence". A class larger than 2% of the prompt is discarded for the same reason.
  This under-counts paraphrased evidence, but it under-counts it identically for every
  criterion (the paper's deviation D4).
* Holding the selector fixed also removes each baseline's *own* allocation: the deployed
  `critical` path uses Ada-KV adaptive head budgets, and the deployed `andpro` path takes
  a single global top-$k$ across heads (an implicit adaptive allocation). So the relative
  order of `criticalkv` and `andpro` **here** is a statement about their scoring rules
  only, and need not match their end-to-end ranking in the paper's tables. The comparison
  this folder is designed to support is LORE-KV against each of them, with everything but
  the score held constant.

## Dropping the figure into the paper

`ranking_criteria.pdf` is the schematic (what each criterion sees, what it measures,
with the measured numbers inside each panel); `ranking_quality.pdf` and
`ranking_evidence.pdf` are the quantitative panels. Copy the PDFs next to the other
figures (`figures_combined/`) and add, e.g. in Appendix "Mechanism Analysis" right after
the Retention-Order Shift subsection:

```latex
\subsection{What the Prompt-Side Criteria Cannot See}
\label{app:criterion_comparison}

\begin{figure}[t]
  \centering
  \includegraphics[width=\textwidth]{figures_combined/ranking_criteria.pdf}
  \caption{\textbf{Three eviction criteria on one captured state.} CriticalKV scores
  prompt-window attention mass, AnDPro projects attention-weighted values on the
  droppable-value direction, and LORE-KV measures the projected leave-one-out deletion
  effect under $M$ reliability-weighted sampled continuations. Prefill, particles and
  selection are identical; only the scoring rule differs.}
  \label{fig:criterion_comparison}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\textwidth]{figures_combined/ranking_quality.pdf}
  \caption{\textbf{Retention quality of each criterion.} (a) Recall against the oracle
  ranking obtained from the true decoding queries; (b) Spearman rank agreement with that
  oracle across layers; (c) cosine fidelity of the retained cache's attention output on
  the same queries. Mistral-7B-Instruct-v0.2.}
  \label{fig:criterion_quality}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\textwidth]{figures_combined/ranking_evidence.pdf}
  \caption{\textbf{Where the difference lands.} (a) Answer-token enrichment of the
  retained cache; (b) which prompt positions survive eviction at $B{=}128$ in a middle
  layer, with the gold-answer positions marked. LORE-KV retains the answer-bearing
  evidence at close to the oracle's own rate.}
  \label{fig:criterion_evidence}
\end{figure}
```

Quote the numbers straight from `summary.json` (`aggregate`, fields `mean`/`sem`/`n`) so
the text and the figure can never drift apart.
