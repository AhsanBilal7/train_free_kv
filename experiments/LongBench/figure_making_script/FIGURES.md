# LORE-KV — mechanism figures: what is measured and how

**One document for both benchmarks.** The nine mechanism figures are produced by the
same code for LongBench and for RULER; only the sample loader and the output folder
differ. This file is therefore the single scientific reference for
`experiments/LongBench/figure_making_script/` **and**
`experiments/RULER/figure_making_script/` (the latter is a set of thin wrappers —
see [§1.3](#13-ruler-is-the-same-code)).

Per figure it gives: the **question** it answers, **where it appears in the paper**,
the **exact quantity the code computes** (not a paraphrase — the formulas below are
what `compute_*.py` runs), what is **plotted**, **why it matters** for the argument,
and the **numbers from the committed runs**, which are the numbers quoted in the
paper text.

Running instructions live in [`README.md`](README.md); implementation deviations
between the paper's equations and the analysis code are collected in
[§5](#5-implementation-notes-and-known-deviations).

---

## Contents

1. [Reproducing the paper's figures](#1-reproducing-the-papers-figures)
2. [The shared capture pipeline](#2-the-shared-capture-pipeline)
3. [Notation and derived quantities](#3-notation-and-derived-quantities)
4. [The ten figures](#4-the-ten-figures)
5. [Implementation notes and known deviations](#5-implementation-notes-and-known-deviations)
6. [How the figures fit together](#6-how-the-figures-fit-together)

---

## 1. Reproducing the paper's figures

### 1.1 Paper ↔ folder map

The paper places LongBench and RULER side by side in one full-width PDF per figure
(`figures_combined/figN_*.pdf`). The scripts here emit **one PNG per benchmark**;
the paper's two-column PDF pairs the LongBench PNG (left) with the RULER PNG (right).

| Folder | Paper figure | Paper location |
|---|---|---|
| `fig1_future_query_alignment/` | `fig:future_query_alignment` | §3.5 Mechanism analysis (main text) |
| `fig2_score_comparison/` | `fig:score_comparison` | App. `app:score_shift` — *How far the retention order actually moves* |
| `fig3_cache_distribution/` | `fig:cache_distribution` | §3.5 Mechanism analysis (main text) |
| `fig4_entropy_trajectory/` | `fig:entropy_trajectory` | App. `app:calibration_extra` — *Trajectory entropy and confidence calibration* |
| `fig5_confidence_calibration/` | `fig:confidence_calibration` | App. `app:calibration_extra` |
| `fig6_mismatch_vs_gain/` | `fig:mismatch_vs_gain` | §3.5 Mechanism analysis (main text) |
| `fig7_retained_token_composition/` | `fig:retained_token_composition` | App. `app:composition` — *What each criterion keeps* |
| `fig8_layer_head_analysis/` | `fig:layer_head_analysis` | App. `app:layer_head` — *Where the future-query signal lives* |
| `fig9_budget_equivalence/` | `fig:budget_equivalence` | App. `app:budget_curve` — *Budget equivalence as a continuous curve* |
| `fig10_particle_divergence/` | `fig:particle_divergence` | App. `app:calibration_extra` — *Trajectory disagreement as the confidence signal* (replaces the entropy term of Eq. 7) |

### 1.2 The configuration the paper's numbers come from

The paper states (App. `app:mechanism_protocol`) that every mechanism figure uses
Mistral-7B-Instruct-v0.2 in the **core** configuration — $M{=}4$, $T{=}8$,
$(\beta,\gamma,\lambda_D){=}(1,1.3,0)$, $\lambda_L{=}1.0$, **uniform head budgets**,
**top-$K$ selection** — at $B{=}128$, on LongBench single-document QA and RULER 4K.

That is `config/lorekv-ent-unif-topk.conf`, and it is the tag on every committed
`.npz` whose numbers appear in the paper. It is **not** the default in
`scripts/run_figures.sh` (which is `lorekv-ent-adap-topk.conf`), so pass it
explicitly:

```bash
# LongBench, single-doc QA  (paper's left column)
CONFIG_FILE=config/lorekv-ent-unif-topk.conf BUDGET=128 SAMPLES=200 \
  BENCHMARK=longbench DATASET=single bash scripts/run_figures.sh

# RULER 4K  (paper's right column)
CONFIG_FILE=config/lorekv-ent-unif-topk.conf BUDGET=128 SAMPLES=200 \
  BENCHMARK=ruler DATASET=4096 bash scripts/run_figures.sh
```

Only three config fields actually reach the figures: `NUM_PARTICLES` ($M$),
`LOOKAHEAD_LEN` ($T$), `LOOKAHEAD_TEMPERATURE`, plus `BETA`, `GAMMA`, `LAMBDA_D`,
`LAMBDA_LOOKAHEAD`, `ROBUST_ALPHA`/`ROBUST_TOPK_BETA` and `CHUNK_SIZE`. The
selection-side switches `BUDGET_TAU`, `LAMBDA_K`, `LAMBDA_V` are inert here because
the figures never call `compress()` — they build retained sets with their own
uniform per-head top-$B$ rule ([§5.3](#53-retention-in-the-figures-is-unchunked-uniform-top-b)).
`LAMBDA_RED` is unused by the `causal_lookahead` code path entirely. This is why
`lorekv-ent-adap-topk` and `lorekv-ent-unif-topk` produce **identical figure
numbers** and differ only in the filename tag.

### 1.3 RULER is the same code

`experiments/RULER/figure_making_script/figN_*/` contains wrappers that inject
`--benchmark ruler --dataset 8192` (override with `--dataset 4096` for the paper's
setting) and `--out-dir <here>`, then call the LongBench module. Everything
scientific — `common.py`, every metric, every plot — is shared. RULER prompts are
built as in `experiments/RULER/pred.build_ruler_input` (chat template over
`context + question`, then the `answer_prefix` token ids appended);
LongBench prompts are built exactly as in `experiments/LongBench/pred.get_pred`
(per-task prompt format, middle-truncation at `--max-length`, `[INST]…[/INST]` or
the Llama-3 chat template except for the six no-chat tasks).

Why both benchmarks: RULER prompts are synthetic and **end with the question
itself**, so the prompt-end window is already a decent stand-in there; LongBench
answers depend on evidence far from the prompt end. An effect that survives on both
is not an artifact of one prompt format — and the *size* of the effect is expected
to be larger on LongBench, which is what Figures 1, 3 and 7 show.

---

## 2. The shared capture pipeline

Every figure is produced from one capture per evaluation prompt, run with the
**actual method code** (`causal_lookahead_snapkv`) so the analysis sees exactly what
eviction sees. `common.collect_and_lookahead()` is a faithful copy of phases 1–2 of
`lookahead_runner.generate_with_lookahead` that **stops before `compress()`**.

**Step (i) — collect prefill.** `lookahead_phase="collect"`; one full uncompressed
forward pass. Each layer's `kv_cluster` retains
the post-RoPE prompt-window queries $\mathcal{Q}^{l,h}_{\mathcal{W}}$ (last $w{=}32$),
and the compact (not GQA-repeated) full $\mathbf{K}^{l,h},\mathbf{V}^{l,h}$, plus
`o_proj_weight`, `head_dim`, `num_key_value_groups`.

**Step (ii) — lookahead particles.** $M$ continuations of length $T$ are sampled
from the shared prompt cache (Eq. (2) `eq:lookahead_cont`), each on its own clone of
the prefill `DynamicCache`. Per particle the code records the query states
$\mathcal{Q}^{(m),l,h}_{\mathrm{look}}$ (onto `kv_cluster._lookahead_buffers[m]`) and,
**separately**, the mean token log-probability and the mean next-token entropy —
separately, so Figures 4 and 5 can ablate the two halves of Eq. (5). The combined
confidence and weights follow Eqs. (5)–(6) exactly:

$$c^{(m)}_u=\log q(y^{(m)}_u\mid h^{(m)}_u)+H\big(q(\cdot\mid h^{(m)}_u)\big),\qquad \mathbb E[c_u\mid h_u]=0,$$
$$R_m=\beta\,\bar c_m-\gamma\,\bar H_m-\lambda_D\,\hat D_m,\qquad \boldsymbol{\omega}=\operatorname{softmax}(\mathbf{R}).$$

The centering is what makes the three channels independent: a sampled token's
log-probability already contains the entropy of the distribution it came from
($\mathbb E[\log q(y_u)\mid h_u]=-H_u$), so adding $H_u$ back leaves only branch
atypicality. Note $\bar c_m$ is recoverable from any existing capture with no
re-decoding, as (mean log-prob) $+$ (mean entropy). There is no $\eta$: it is not
identifiable alongside $(\beta,\gamma,\lambda_D)$.

**Step (iii) — the ground truth $\mathcal{Q}_{\mathrm{future}}$.** The model
**greedy-decodes** the answer against the **full, uncompressed** cache and every
post-RoPE query state it produces is recorded. This is the only part of the capture
that does not exist at inference time; it is what makes the mechanism claims
measurable rather than asserted (App. `app:kv_eviction_problem`: $\mathcal{Q}_{\mathrm{future}}$
is unknown *at eviction time* but perfectly observable *after* the answer exists).

Capture is **non-invasive**: `common.PostRoPEQueryCapture` registers forward
*pre*-hooks that read `hidden_states` and `position_ids` and recompute
$\mathbf{q}$ from the layer's own `q_proj` + `rotary_emb`. The forward pass itself is
untouched, so step (iii) can never influence steps (i)–(ii). It runs **before** the
particles and clears the buffers it incidentally polluted
(`reset_lookahead_buffers()`), so the particles are unaffected.

**Scoring.** Every figure that needs an importance score calls each layer's own
`TestTimeLookaheadKVCluster._project_and_score(key, val, query_batches, causal_flags)`
— the production scorer, not a reimplementation. Given GQA-repeated $K,V$ and the
batch list `[window, particle_0 … particle_{M-1}]` (`causal_flags=[True, False…]`) it
returns one $[H,n]$ score tensor per batch. Internally, per head:

$$a_{q,i}=\operatorname{Softmax}_i\!\Big(\tfrac{q^\top k_i}{\sqrt d}\Big),\quad
 y_q=\textstyle\sum_j a_{q,j}v_j,\quad
 c_{q,i}=\tfrac{a_{q,i}}{1-a_{q,i}+\epsilon},\quad
 s_{q,i}=|c_{q,i}|\cdot\lVert W^O v_i - W^O y_q\rVert_2,$$

then aggregated over the query batch by $\rho(\cdot)=\alpha\operatorname{TopAvg}_\beta+(1-\alpha)\operatorname{Mean}$
(Eq. (19) `eq:robust_rho`; $\alpha{=}0,\beta{=}1$ in the shipped configs → plain
mean, the core method). Note the norm is **un-squared** — see [§5.1](#51-the-score-is-an-unsquared-norm).
The causal flag matters: window queries must not attend to *later* window positions;
lookahead and future queries postdate the whole prompt, so no mask.

**Robustness of the run.** Each finished sample is fully released before the next
(`common.free_sample`); a sample too large for the GPU is skipped with an OOM guard
rather than crashing the run; `FIG_MAX_ANALYSIS_TOKENS` (24000 in the launchers)
middle-truncates over-long prompts so they still contribute. Every `compute_*.py`
checkpoints after each sample to scratch and resumes on relaunch. Consequence for
reading the figures: the effective sample count is recorded in the `.npz`
(`n_samples`, or the array lengths) and printed in the plot title — it can be lower
than `SAMPLES × #datasets`.

---

## 3. Notation and derived quantities

Symbols follow the paper; the code name is given where it differs.

| Symbol | Meaning |
|---|---|
| $X$, $n$, $B$, $l,h$ | prompt, prompt length, KV budget, layer / KV head |
| $w{=}32$ | prompt-end observation window |
| $\mathcal{Q}^{l,h}_{\mathcal{W}}$ | post-RoPE prompt-window queries — what SnapKV / AnDPro score with |
| $\mathcal{Q}^{(m),l,h}_{\mathrm{look}}$ | post-RoPE queries of lookahead particle $m$ ($m{=}1..M$, length $T$) |
| $\mathcal{Q}_{\mathrm{future}}$ | **true future queries**: post-RoPE queries from greedy-decoding the answer against the full cache (code: `true_queries` / `Q_true`) |
| $R_m,\ \omega_m$ | particle confidence and normalized weight, Eqs. (5)–(6) |
| $\bar H_m$ | that particle's mean **next-token** entropy — *not* semantic entropy |
| $e^{l,h}_{q,i}$ | projected leave-one-out perturbation, Eqs. (10)–(11) |
| $S^{l,h}_i$ | token score, Eq. (13): $\rho(\mathcal{E}_{i,\mathcal W})+\lambda_L\sum_m\omega_m\rho(\mathcal{E}_{i,m})$ |
| $\mathcal{P}$ | protected set $\{1\}\cup\mathcal{W}$ (structural markers $\mathcal{B}$ are empty here) |

Three quantities recur across figures and are defined once in `common.py`:

**Cosine coverage** (`weighted_coverage`, Eq. (23) `eq:coverage`) — how well an
estimated query set $\hat{\mathcal Q}$, split into weighted sources $s$, matches the
true future queries:

$$\operatorname{Cov}\big(\mathcal Q_{\mathrm{future}};\{\hat{\mathcal Q}_s\},w\big)=\frac1H\sum_h\frac1{T_{\mathrm f}}\sum_t\sum_s w_s \max_{\hat q\in\hat{\mathcal Q}_s}\cos\big(q^{\mathrm{future}}_{t,h},\hat q\big).$$

Read it as: *for every query the model really used, how close was the nearest
estimated query* — averaged over heads and decoding steps. A single source with
$w{=}1$ recovers plain max-cosine coverage; weighting sources is what lets a
confident-but-wrong particle be down-weighted. Everything is unit-normalized in
float32 first. The per-sample value is then averaged over layers.

**Oracle score and Recall@B** (`oracle_scores`, `recall_at_b`) — $S^\star$ is
Eq. (13)'s perturbation score computed **under $\mathcal Q_{\mathrm{future}}$
itself**: the importance an eviction method *would* assign if it could see the
future. Then

$$\operatorname{Recall}@B(S,S^\star)=\frac{\lvert\mathrm{top}_B(S)\cap\mathrm{top}_B(S^\star)\rvert}{B},$$

the fraction of the ideal retention a proxy score recovers.

**Retained mask and fidelity** (`head_topb_mask`, `attn_outputs`) — a score $[H,n]$
becomes a retained set by forcing $\mathcal{P}$ in (position $0$ and the last $w$,
score $\to+\infty$) and taking per-head top-$\min(B,n)$. Behavioural fidelity is then

$$F=\mathbb{E}_{q\sim\mathcal Q_{\mathrm{future}}}\cos\big(\mathbf o^{\mathcal R}_q,\ \mathbf o_q\big),\qquad
\mathbf o^{\mathcal R}_q=\operatorname{Softmax}\big(\text{masked logits}\big)V,$$

i.e. exactly the behavioural objective of App. `app:kv_eviction_problem` that the
eviction score is a proxy for.

**The three criteria compared throughout.** Every figure that contrasts methods uses
the same three, all built from one capture so they differ *only* in the criterion:

| Name in code | What it is | Stands for |
|---|---|---|
| `attention` | mean prompt-window attention weight $\bar a_i$ | SnapKV / H2O-style attention-only |
| `andpro_window` / `s_win` | Eq. (13) with $\lambda_L{=}0$ — projected perturbation under window queries only | AnDPro-style prompt-side output-aware |
| `lorekv_multi` / `s_multi` | full Eq. (13), window $+$ $M$ confidence-weighted particles | LORE-KV |

`single_lookahead` (window $+$ **one** particle, unweighted) appears in Figure 2 and
stands for Lookahead Q-Cache.

---

## 4. The ten figures

### Figure 1 — Future-query alignment (`fig1_future_query_alignment/`)

**Question.** Do confidence-weighted *multiple* lookahead trajectories approximate
the queries that actually drive decoding better than the prompt window or a single
trajectory?

**Computed.** One capture per sample with $M=\max(\texttt{--m-list})=8$ particles and
`--decode-steps 32`. For each $M\in\{1,2,4,8\}$ the code re-weights the **first $M$**
of those particles and evaluates $\operatorname{Cov}(\mathcal Q_{\mathrm{future}};\cdot)$
for five estimators, averaged over layers, then over samples:

| Estimator | Sources | Weights |
|---|---|---|
| `prompt_window` | $\{\mathcal{Q}_{\mathcal W}\}$ | $1$ |
| `single_traj` | $\{\mathcal{Q}^{(1)}_{\mathrm{look}}\}$ | $1$ |
| `uniform` | $M$ particles | $1/M$ |
| `logprob` | $M$ particles | $\operatorname{softmax}(\eta\,\mathbf{logp})$ |
| `lorekv` | $M$ particles | $\operatorname{softmax}(\bar{\mathbf c}-\gamma\bar{\mathbf H})$ at $\gamma=\gamma_E$ — Eq. (6) |

The $M$-sweep re-uses **nested prefixes of one 8-particle draw**, not independent
re-samples; it is therefore a within-sample refinement curve, and the $M{=}1$ column
is identical across the three multi-particle estimators by construction.

**Plotted.** Coverage vs $M$ (log$_2$ abscissa); the two $M$-independent estimators
are drawn as horizontal reference lines; bands are $\pm1$ s.d. over samples (the
paper's caption reports them as s.e.m. — see [§5.7](#57-plot-level-caption-mismatches)).

**Why it matters.** The **central-premise validation**. The whole method rests on the
claim that prompt-side queries are the wrong reference; this measures it directly.
Two readings carry the argument: (a) the large step is from *input-side* to
*response-side* queries, which is exactly the substitution §2.2 makes; (b) the
residual benefit of more particles **saturates by $M\approx4$** — a criterion effect,
not a compute effect, since extra compute would keep paying off.

**As run** (`lorekv-ent-unif-topk`, $B{=}128$, $T{=}8$):

| | prompt window | single traj | LORE-KV $M{=}2$ | $M{=}4$ | $M{=}8$ | $n$ |
|---|---|---|---|---|---|---|
| LongBench single-doc QA | 0.7882 | 0.8256 | 0.8278 | 0.8279 | 0.8285 | 118 |
| RULER 4096 | 0.7790 | 0.7842 | 0.7842 | 0.7841 | 0.7841 | 200 |

The gap is wide on LongBench and narrow on RULER, exactly as the prompt formats
predict.

---

### Figure 2 — Score comparison / retention shift (`fig2_score_comparison/`)

**Question.** Does the lookahead term change *which* tokens are kept, or does it only
re-scale the prompt-side score?

**Computed.** No true-query capture is needed (`--decode-steps 0`). Per layer, three
head-averaged per-token scores over the same candidate set:

- `attention` — `attention_score_layer()`: softmax attention of the $w$ window
  queries against all $n$ keys, **with** the causal mask on the trailing $w\times w$
  block, averaged over window positions and then over heads.
- `single_lookahead` — $s_{\mathcal W}+\lambda_L\,s_{(1)}$.
- `lorekv_multi` — $s_{\mathcal W}+\lambda_L\sum_m\omega_m s_{(m)}$ (Eq. (13)).

For each of the three pairs it records Spearman $\rho$ and top-$B$ overlap
$|\mathrm{top}_B(a)\cap\mathrm{top}_B(b)|/B$, accumulated over all layers and samples.
Separately it stores the full per-token profile of one **representative layer**
(default: middle layer) from the **first** sample, for the profile panel.

**Plotted.** Left — rank-ordered, min–max-normalized score profiles of the
representative layer on a log abscissa, with a dotted line at $B$. Right — grouped
bars of top-$B$ overlap and Spearman per pair (mean $\pm$ s.d.).

**Why it matters.** Guards against the "it's just re-weighted attention" objection.
High rank correlation with imperfect top-$B$ overlap is the expected signature: the
rules agree about the bulk ordering and disagree **at the boundary**, and at a tight
budget the boundary is all that matters.

**As run** ($B{=}128$):

| pair | LongBench overlap / $\rho$ | RULER overlap / $\rho$ |
|---|---|---|
| attention vs single-lookahead | 0.821 / 0.939 | 0.906 / 0.913 |
| attention vs LORE-KV multi | 0.823 / 0.941 | 0.861 / 0.873 |
| single-lookahead vs LORE-KV multi | 0.972 / 0.996 | 0.919 / 0.950 |

≈ one retained token in five differs between attention and LORE-KV at $B{=}128$,
while the two lookahead rules agree far more with each other than either does with
attention — consistent with Figure 1's "the big step is response-side queries".

---

### Figure 3 — Full-cache vs evicted-cache output (`fig3_cache_distribution/`)

**Question.** At the *same* budget, does the LORE-KV cache reproduce what the model
actually computes at decode time better than attention-only eviction?

**Computed.** Per layer, with `--decode-steps 16` true queries $q$: build
`mask_lorekv` from $s_{\text{multi}}$ and `mask_attn` from the window-attention
score, both via `head_topb_mask` at the same $B$ and the same protected set. Then
compute $\mathbf o_q$ on the full cache and $\mathbf o^{\mathcal R}_q$ under each
mask, and record **per (layer, head, query)**:

$$\cos\big(\mathbf o^{\mathcal R}_q,\mathbf o_q\big)\quad\text{and}\quad
\frac{\lVert \mathbf o^{\mathcal R}_q-\mathbf o_q\rVert_2}{\lVert \mathbf o_q\rVert_2}.$$

Both the full pooled distribution and a per-layer mean are saved.

**Plotted.** Left — density of the cosine (means in the legend). Right — the same
cosine averaged per layer across depth.

**Why it matters.** Connects the eviction *criterion* to the quantity the paper
claims to preserve — the residual-stream output, i.e. the behavioural objective of
App. `app:kv_eviction_problem` that Eq. (11) is designed to protect. The interesting
difference is **in the low-fidelity tail**: those are the cases where eviction
discarded something a query actually needed.

**As run** ($B{=}128$): LongBench mean cosine **0.873 → 0.905**; RULER 4096
**0.929 → 0.944** (attention-only → LORE-KV). The per-layer panel shows LORE-KV at or
above the attention-only criterion at essentially every depth, so this is not a
single-layer artifact.

---

### Figure 4 — Trajectory entropy (`fig4_entropy_trajectory/`)

**Question.** Why sample *several* trajectories and down-weight uncertain ones
instead of trusting one draft?

**Computed.** $M{=}8$ particles, `--decode-steps 32`. Per particle: its mean
next-token entropy $\bar H_m$ and its **single-particle coverage**
$\operatorname{Cov}(\mathcal Q_{\mathrm{future}};\{\mathcal Q^{(m)}_{\mathrm{look}}\})$,
layer-averaged. Per sample: coverage under `single_min` / `single_mean` /
`single_max` over particles, and under `uniform` / `logprob` / `lorekv` weighting of
all $M$.

Note the fallback `gamma_E = config.particle_gamma if > 1.0 else 1.3` — with a
$\gamma\le1$ config this figure still uses $1.3$, so it always shows the
entropy-weighted cell the paper describes. In the reparametrized coordinates
`logprob` is $(\beta,\gamma,\lambda_D)=(1,1,0)$ and `lorekv` is $(1,\gamma_E,0)$;
note that `logprob` therefore carries a *full unit* of entropy penalty and is **not**
a no-entropy control.

**Plotted.** Left — scatter of $\bar H_m$ vs single-particle coverage with a
least-squares fit and Pearson $r$. Right — bars comparing a single draw
(worst / mean / best) against the three combination rules.

**Why it matters.** Supplies the premise Eq. (5) rests on: if an uncertain
continuation were *not* a worse query reference, the entropy penalty would have no
reason to exist. It also makes the variance argument concrete — a single draft is a
lottery whose usefulness depends on which branch it fell into, and entropy is an
observable, training-free indicator of having drawn a bad one.

**As run.** LongBench: $r=-0.22$ over 2400 particles — clearly negative. On RULER
nearly all particles are near-deterministic, so the entropy range is tiny and the
slope is correspondingly flat ($r\approx-0.04$ in the paper's run); that flatness is
a property of the benchmark, not a failure of the signal.

*Scope note:* the paper's caption for this figure describes only the scatter panel;
the bar panel is extra diagnostic the repo produces and the paper does not show.

---

### Figure 5 — Confidence calibration (`fig5_confidence_calibration/`)

**Question.** Does the combined confidence $R_m$ rank particles by how *useful* they
are, or merely by how *fluent* they are?

**Computed.** $M{=}8$ particles, `--decode-steps 32`. Per particle, two oracle
utilities, both layer-averaged:

- `util_align` — coverage of $\mathcal Q_{\mathrm{future}}$ by that particle's queries alone.
- `util_recall` — $\operatorname{Recall}@B\big(s_{(m)},S^\star\big)$, i.e. how much of
  the oracle retention that one particle's own perturbation score recovers.

and five candidate signals: mean log-probability, mean entropy, the trajectory
disagreement $\hat D_m$ that **replaced** entropy in Eq. (7), combined $R_m$, and the
normalized weight $\omega_m$. The plot stage relates signals to utility
(`--utility recall` by default). Figure 10 is the dedicated treatment of $\hat D_m$;
it appears here so all confidence signals are ranked on one common axis.

**Plotted.** (A) calibration — particles binned into equal-count bins by normalized
$R_m$ vs observed utility (mean $\pm$ s.d.); (B) risk–coverage — keep only the most
confident fraction, plot residual coverage error $1-\operatorname{Cov}$ (this panel
always uses `util_align`); (C) Spearman of each signal against utility.

**Why it matters.** This is the figure that keeps the confidence weighting honest.
It supports the design *and* bounds it: confidence carries real signal, so tilting
the weights is justified — but the signal is **weak**, so the weighting must stay
soft ($\eta$ finite, no hard top-particle selection) and enter with a bounded
$\lambda_L$, which are exactly the safeguards of App. `app:confidence_details`. It
also explains why the $\gamma{=}1$ ablation row sits close to the core
configuration rather than far below it: the entropy penalty refines the weighting,
it is not the source of the gain.

**As run** (Spearman vs `util_recall`, $B{=}128$, $(\beta,\gamma,\lambda_D){=}(1,1.3,0)$):

| signal | LongBench | RULER 4096 |
|---|---|---|
| log-prob | 0.055 | 0.036 |
| $-$entropy | 0.016 | 0.025 |
| combined $R_m$ | 0.043 | 0.027 |
| weight $\omega_m$ | 0.008 | $-0.029$ |

All small ($\rho\le0.06$), log-probability the strongest single signal — which is the
paper's "real but weak" reading, and the risk–coverage curve is correspondingly flat.

---

### Figure 6 — Mismatch vs gain (`fig6_mismatch_vs_gain/`)

**Question.** Does LORE-KV help most on exactly the examples where the prompt window
is least representative of that example's own future queries?

**Computed.** `--decode-steps 24`, ≥8 samples per dataset. Per example, all
layer-averaged:

$$\operatorname{Mis}(X)=1-\operatorname{Cov}\big(\mathcal Q_{\mathrm{future}};\{\mathcal Q_{\mathcal W}\}\big)$$

$$\text{recall gain}=\operatorname{Recall}@B(S_{\text{multi}},S^\star)-\operatorname{Recall}@B(S_{\mathcal W},S^\star)$$

$$\text{fidelity gain}=\cos(\mathbf o^{\mathcal R_{\text{LORE}}},\mathbf o)-\cos(\mathbf o^{\mathcal R_{b}},\mathbf o),\quad b\in\{\text{window-only},\ \text{attention}\}$$

(the symbol is $\operatorname{Mis}$, not $M$, because $M$ is the particle count
everywhere else.)

**Plotted.** One marker per example; least-squares fit with slope and Spearman $\rho$
printed. Left panel uses the recall gain, right the fidelity gain **vs attention**.

**Why it matters.** The strongest and most causal evidence in the set. It ties the
**diagnosed problem** to the **observed benefit** example by example rather than on
average — and it is the observation a generic "more compute helps" account cannot
produce, since extra compute would lift all examples roughly equally, whereas fixing
a query-reference error should help most where that error is largest.

**As run** ($B{=}128$, $M{=}4$): every slope positive on both benchmarks —
LongBench $\rho=0.28$ (recall) / $0.53$ (fidelity) over 265 examples;
RULER 4096 $\rho=0.57$ / $0.51$ over 200 examples.

---

### Figure 7 — Retained-token composition (`fig7_retained_token_composition/`)

**Question.** *What kinds* of tokens does each criterion actually keep?

**Computed.** No true-query capture (`--decode-steps 0`). Every prompt token is
labelled with one class, in this **priority order** (`classify_tokens`):
`sink` (position 0) → `recent` (last $w$) → `answer/evidence` (the token lies inside
an exact token-id match of a gold answer string, tried as `ans`, `" "+ans`,
`ans.strip()`) → `number` (contains a digit) → `punct/boiler` (no alphanumeric
character) → `entity(cap)` (capitalized, all-alphabetic) → `other`.

Then, per layer, retained masks are built for all three criteria at the same $B$, and
per token the **fraction of heads that keep it** is accumulated and averaged over
layers. Per class $c$ the figure reports

$$\text{retention rate}(c)=\underset{i:\,\mathrm{class}(i)=c}{\mathrm{mean}}\ \underset{h}{\mathrm{mean}}\ \mathbb 1[i\in\mathcal R^{l,h}],
\qquad
\text{enrichment}(c)=\frac{P(c\mid\mathcal R)}{P(c\mid X)}.$$

**Plotted.** Grouped bars, retention rate and enrichment per class per method, with a
dotted line at enrichment $=1$.

**Why it matters.** Interpretability plus a concrete failure-mode check: aggressive
eviction can remove precisely the evidence future queries need. Two classes are
**uninformative by construction** — `sink` and `recent` are in $\mathcal P$, so every
method keeps them and every method shows the same large enrichment; they are shown
only for completeness. The informative class is `answer/evidence`.

**As run** ($B{=}128$), `answer/evidence` retention / enrichment:

| | attention | AnDPro-window | LORE-KV |
|---|---|---|---|
| LongBench | 0.028 / 1.03 | 0.027 / 1.00 | **0.068 / 2.51** |
| RULER 4096 | 0.549 / 16.51 | 0.542 / 16.28 | **0.677 / 20.34** |

On LongBench the two prompt-side criteria retain answer-bearing tokens at essentially
the base rate of the prompt — no better than chance at protecting the evidence — while
LORE-KV retains them ≈2.4× more often. The remaining classes move very little, so the
extra evidence retention is bought at the margin, not by a wholesale change in what
the cache looks like. RULER's `number` class is empty under this labelling (its
digits are inside the planted needle and get the higher-priority `answer/evidence`
label), which is why the paper notes that group as empty.

*Caveat to state when citing this figure:* `answer/evidence` is an **exact token-id
subsequence match** of the gold answer. Paraphrased or reformulated answers are
undercounted, so the class is a lower bound on true evidence — which makes the
comparison across methods conservative but the absolute rate an underestimate.

---

### Figure 8 — Layer / head structure (`fig8_layer_head_analysis/`)

**Question.** Where in the network does the future-query signal live?

**Computed.** `--decode-steps 24`. Three accumulators:

- **Layer perturbation energy** $E_l=\sum_h\sum_i S^{l,h}_i$ — the code uses
  `s_multi.clamp_min(0).sum()` over **all** $n$ positions (see [§5.4](#54-e_l-sums-over-all-positions)),
  averaged over samples. This is the mass the adaptive head-budget plug-in
  (Eq. (21) `eq:adaptive_budget`) would allocate on.
- **Per-head recall gain** $G_{l,h}=\operatorname{Recall}@B(S^{\mathrm{multi}}_h,S^\star_h)-\operatorname{Recall}@B(S^{\mathcal W}_h,S^\star_h)$
  — computed **per head**, not on head-averaged scores.
- **Per-head mismatch** $MM_{l,h}=1-\operatorname{Cov}$ of window vs true queries for
  that head.

**Plotted.** Left — $E_l$ across depth. Right — heatmap of $G_{l,h}$ over
layers × heads on a symmetric diverging scale (red = gain, blue = loss, white = no
change); `--heatmap mismatch` swaps in $MM_{l,h}$.

**Why it matters.** Two conclusions. *Interpretively*, the future-query signal is a
property of particular heads rather than a diffuse effect, consistent with the
head-specialization observations that motivate head-wise retention policies.
*Practically*, this heterogeneity is the premise the $\tau>0$ branch of
$\operatorname{Alloc}$ rests on — and if the adaptive-budget ablation yields only
small accuracy changes, this figure is what says the core score is already capturing
most of what reallocation could recover, rather than that the heterogeneity is absent.

---

### Figure 9 — Budget equivalence (`fig9_budget_equivalence/`)

**Question.** How much cache does LORE-KV save at equal behavioural fidelity?

**Computed.** One capture per sample (`--decode-steps 24`); scores are computed
**once** and then swept over $B\in\{64,128,256,512,1024\}$ — valid because the
perturbation score does not depend on the budget, only the selection does. (Hence
`--score-budget 512` merely configures the model.) For each method and budget,
$F(B)=\mathbb{E}_q\cos(\mathbf o^{\mathcal R(B)}_q,\mathbf o_q)$ averaged over layers,
heads, queries and samples. The plot stage inverts the baseline curves by linear
interpolation:

$$B^{\mathrm{eq}}_b(B)=\min\{B' : F_b(B')\ge F_{\text{LORE-KV}}(B)\},\qquad \text{saving}=1-\frac{B}{B^{\mathrm{eq}}_b(B)}.$$

**Plotted.** Left — $F(B)$ per method (log$_2$ abscissa). Right — saving (%) vs
LORE-KV budget, one curve per baseline; a **missing point** means the baseline never
reaches that fidelity anywhere in the sweep, so no equivalent budget exists and the
saving there is a lower bound, not a gap that closed.

**Why it matters.** Restates the win in the currency a deployment cares about —
memory at equal behaviour — and upgrades the paper's discrete budget-equivalence
*table* into a continuous *curve* with a metric that is not hostage to a single
accuracy threshold. It should be read as a complement to that table, not a
replacement: this is a saving in **cache fidelity**, not directly in task score.

**As run** — $F(B)$:

| $B$ | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| LongBench — attention | 0.818 | 0.867 | 0.911 | 0.948 | 0.975 |
| LongBench — AnDPro-window | 0.818 | 0.866 | 0.909 | 0.945 | 0.973 |
| LongBench — **LORE-KV** | **0.852** | **0.899** | **0.934** | **0.961** | **0.981** |
| RULER — attention | 0.900 | 0.934 | 0.954 | 0.971 | 0.986 |
| RULER — AnDPro-window | 0.902 | 0.935 | 0.956 | 0.973 | 0.987 |
| RULER — **LORE-KV** | **0.916** | **0.943** | **0.962** | **0.978** | **0.989** |

Resulting savings: **33–43 %** on LongBench and **27–33 %** on RULER, sustained across
the sweep rather than at one operating point (no equivalent budget exists at
$B{=}1024$ on either benchmark). Note the two prompt-side criteria are essentially
tied with each other on both benchmarks.

---

### Figure 10 — Trajectory disagreement (`fig10_particle_divergence/`)

**Question.** The second term of the particle confidence $R_m$ (Eq. 7) used to be an
absolute predictive-entropy penalty. It is now the particle's **trajectory
disagreement** with the rest of the ensemble. Is that the better signal, and does the
weighting built on it actually help?

**The quantity.** For particle $m$,

$$D_m \;=\; \frac{1}{M-1}\sum_{r\neq m}\;\frac{1}{T|\mathcal G|}\sum_{u,g}
\Big[\,1-\cos\big(q^{(m)}_{u,g},\,q^{(r)}_{u,g}\big)\Big],$$

$u$ indexing the $T$ lookahead steps and $g$ the (layer, head) groups. Matched queries
sit at the same absolute position $n+u$, so they carry the same rotary phase and the
cosine is directly comparable. $D_m$ is standardized across the $M$ particles,
$\hat D_m=(D_m-\bar D)/\mathrm{sd}(D)$, before entering $R_m$.

**Why it replaces entropy.** The particles are themselves drawn from $p_\theta$, so
$\mathbb E[\log p_\theta(y_u)\mid y_{<u}]=-H(p_\theta(\cdot\mid y_{<u}))$: in
expectation the entropy term re-counts the log-probability term, and the softmax tilt
then targets a likelihood-tilted (mode-seeking) distribution rather than $p_\theta$.
$\hat D_m$ instead is *ensemble-relative* and sums to zero over the particles, so the
tilt only re-allocates mass **within** the sample from $p_\theta$; and it is measured
in the space the eviction score actually consumes (query directions) rather than in
token-likelihood space. It also covers the failure mode App. `app:limitations` names
— a confident but wrong trajectory keeps a high entropy-based confidence, but is
down-weighted as soon as no other particle reproduces its queries.

**Computed.** $M{=}8$ particles, `--decode-steps 32`. Per particle: $\log p$,
$\bar H_m$, $D_m$, $\hat D_m$, and the same two oracle utilities as Figure 5
(`util_align`, `util_recall`). Per sample: coverage of $\mathcal Q_{\mathrm{future}}$
under four weightings of the **identical** particles — uniform, log-prob,
$\bar c-\gamma\bar H$ (the published rule), and
$\log p-\lambda_D\hat D$ (this method). Since every rule sees the same particles, no
extra test-time compute enters the comparison and any difference is attributable to
the weights alone.

**Plotted.** (A) $D_m$ vs oracle utility with fit, Pearson $r$ and Spearman $\rho$;
(B) the same particles against $\bar H_m$, so the two signals can be read side by
side; (C) Spearman of every signal against utility, sign-oriented so higher = better
particle; (D) ensemble coverage per weighting rule ($\pm1$ s.e.m., since the bars
compare means and the across-prompt spread dwarfs the between-rule difference), with
the **paired** per-sample mean difference and win rate against the entropy rule
printed above.

**How to read it.** Panel A is the premise: a downward slope means an atypical
trajectory is a worse proxy for $\mathcal Q_{\mathrm{future}}$. Panel C is the
comparison that matters — if $-\hat D_m$ ranks above $-\bar H_m$ there, the
replacement is carrying information entropy did not. Panel D is the consequence.
Expect the coverage differences to be small in absolute terms (Figure 5's lesson that
the confidence signal is real but weak applies here too), which is why the weighting
stays soft and $\lambda_D$ is bounded; the paired win rate, not the bar heights, is
the statistic to quote.

**Run it.**

```bash
FIGURES=10 CONFIG_FILE=config/lorekv-dis-unif-topk.conf BUDGET=128 \
  BENCHMARK=longbench bash scripts/run_figures_aces.sh
```

---

## 5. Implementation notes and known deviations

These are places where the analysis code and the paper's equations do not line up
exactly. None of them changes a figure's direction, but each should be stated
correctly if a number is quoted or a reviewer reads the code.

### 5.1 The score is an unsquared norm

Eq. (11) `eq:projected_score` writes $e^{l,h}_{q,i}=\lVert W^{O}\Delta\mathbf o\rVert_2^2$.
`_project_and_score` returns $|c_{q,i}|\cdot\lVert W^O v_i-W^O y_q\rVert_2$ — the
**un-squared** norm. Per single query this is a monotone transform, so the *ranking*
and hence the retained set are unchanged; but it is **not** equivalent after
averaging over a query set, and it changes the numeric scale of $E_{l,h}$
(Figure 8) and of the head-budget mass in Eq. (21). Either square the returned score
or drop the exponent in Eq. (11) — the paper and the code should say the same thing.

### 5.2 Two attention baselines that differ slightly

Figure 2's `attention_score_layer` applies the causal mask to the trailing
$w\times w$ block before the softmax; Figures 3, 6, 7 and 9 compute
`softmax(q_W Kᵀ/√D).mean(1)` **without** that mask. The difference is confined to how
much mass the last $w$ positions receive — and those positions are protected in every
retained set anyway — but the two are not literally the same baseline.

### 5.3 Retention in the figures is unchunked, uniform top-$B$

Production eviction (`_select_chunks`) selects **chunks of size 4** by max-pooled
chunk score, applies adaptive head budgets ($\tau$) and optionally the diversity
penalty. The figures use `head_topb_mask`: per-token, per-head top-$\min(B,n)$ with
$\mathcal P$ forced in, uniform across heads, no chunking, no diversity. This is
deliberate — it isolates the **criterion** from the selection machinery, so the three
methods are compared under an identical selection rule — but it means the figures
measure the score, not the deployed retained set. It is also why `BUDGET_TAU`,
`LAMBDA_K`, `LAMBDA_V` have no effect on any figure. Note also that $B$ here is the
per-head total **including** the ~33 protected positions.

### 5.4 $E_l$ sums over all positions

The paper defines $E_{l,h}=\sum_{i\in\mathcal C}S^{l,h}_i$ (candidates only);
`compute_layer_head.py` sums `s_multi.clamp_min(0)` over all $n$ positions, protected
ones included. The depth profile is unaffected in shape but the absolute values
include the protected tail.

### 5.5 $\mathcal Q_{\mathrm{future}}$ is one greedy continuation

The ground truth is the queries produced by **greedy** decoding against the full
cache (32 steps for Figs. 1/4/5, 24 for Figs. 6/8/9, 16 for Fig. 3). A different
decoding temperature would give a somewhat different reference set, so every coverage
number measures agreement with **one canonical continuation**, not with the full
continuation distribution. This is stated as a limitation in App. `app:limitations`
and should not be quietly dropped when the figures are cited.

### 5.6 Particle counts and $\gamma$ differ by figure

Figures 1, 4, 5 and 10 override $M$ ($8$, $8$, $8$, $8$) to have enough particles for
their sweeps; Figures 2, 3, 6, 7, 8, 9 use the config's $M{=}4$. Figures 4, 5 and 10
substitute $\gamma{=}1.3$ if the config sets $\gamma\le1$, and Figures 1 and 10
likewise substitute $\lambda_D{=}0.3$ if the config sets $\lambda_D{=}0$, so both
weighting rules stay comparable whichever config drives the run. Figure 1's $M$-sweep
uses nested prefixes of a single 8-particle draw rather than independent re-samples;
because $D_m$ is ensemble-relative it is **recomputed within each prefix** rather than
sliced from the full-ensemble value.

### 5.7 Plot-level caption mismatches

The paper's combined PDFs are assembled outside this repo and use log ordinates in
places where the committed `plot_*.py` uses linear ones: Fig. 2 (score profiles),
Fig. 3 (cosine density) and Fig. 7 (both ordinates) are described in the captions as
logarithmic. Fig. 2's caption also says each profile is normalized "by its own
maximum" while `plot_scores.py` applies min–max normalization. Fig. 1's caption calls
the bands $\pm1$ s.e.m. while `plot_alignment.py` draws $\pm1$ s.d.; Fig. 3's caption
says means are printed inside the panel while the script puts them in the legend.
Either regenerate the paper figures from these
scripts with the axes/bands adjusted, or reconcile the captions — as it stands the
committed PNGs and the paper PDFs are not pixel-equivalent.

### 5.8 Sample counts are effective, not nominal

OOM skips and `FIG_MAX_ANALYSIS_TOKENS` truncation mean the realized $n$ can be below
`SAMPLES × #datasets` (e.g. the committed LongBench Fig. 1 has $n{=}118$ across three
tasks). The realized count is in the `.npz` and in the plot title; quote that one.

---

## 6. How the figures fit together

- **1, 4** establish the *input* is right — the estimated query set approximates the
  queries the cache will actually serve, and uncertain trajectories are the ones worth
  down-weighting.
- **2, 3** establish the *criterion* is right — it reorders importance at the budget
  boundary (2) and the resulting cache reproduces the full-cache output (3).
- **5** bounds the *weighting* that combines them: real signal, weak signal, hence
  soft and capped.
- **6** links the diagnosed mismatch to the measured gain example by example — the
  causal evidence, and the one an "extra compute helps" account cannot reproduce.
- **7, 8** interpret *what* survives (answer/evidence tokens) and *where* the effect
  lives (specific heads, mid-to-late layers).
- **9** cashes the result out as cache saved at equal behaviour.

Figures 1, 3 and 6 carry the main-text argument; 2, 4, 5, 7, 8, 9 are the appendix
checks that each main-text claim leans on.
