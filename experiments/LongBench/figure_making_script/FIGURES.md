# LORE-KV / TLC-KV — figure reference

One entry per figure: the **question** it answers, the **formula** behind it, what
is **plotted**, and **why it matters** for the paper's argument. Notation is shared
across figures and defined once below. (Usage/commands live in `README.md`; this
document is the scientific rationale.)

## Shared notation

- $X$ — prompt; $n$ — prompt length; $B$ — KV budget; $l,h$ — layer / KV head.
- $\mathcal{Q}_W$ — post-RoPE **prompt-window** queries (last $w{=}32$).
- $\mathcal{Q}^{(m)}_{\text{look}}$ — post-RoPE queries of **lookahead particle** $m$ ($m=1..M$, length $T$).
- $\mathcal{Q}_{\text{true}}$ — the **true future queries**: post-RoPE queries produced by greedy-decoding the answer against the **full (uncompressed) cache**. This is the reference every proxy is trying to match.
- **Confidence** (Eq. 5–6): $R_m=\frac1T\sum_t\log p_\theta(y^{(m)}_t\mid X,y^{(m)}_{<t})-\lambda_H\frac1T\sum_t H\!\big(p_\theta(\cdot\mid X,y^{(m)}_{<t})\big)$, weights $\omega=\mathrm{softmax}(\eta R)$. *$H$ is next-token entropy, not semantic entropy.*
- **Perturbation score** (Eq. 10–11): for query $q$ and candidate $i$, $\;\Delta o_{q,i}=\frac{a_{q,i}}{1-a_{q,i}+\epsilon}\,(v_i-o_q)$, $\;e_{q,i}=\lVert W_O\,\Delta o_{q,i}\rVert_2^2$, aggregated over a query set by $\rho(\cdot)$.
- **Token score** (Eq. 13): $S_i=\rho(E_{i,\mathcal W})+\lambda_L\sum_{m}\omega_m\,\rho(E_{i,m})$.
- **Cosine coverage** (how well an estimated query set $\hat{\mathcal Q}$, split into weighted sources $s$, covers the true queries):
  $$\mathrm{Cov}(\mathcal Q_{\text{true}};\{\hat{\mathcal Q}_s\},w)=\frac1{H}\sum_{h}\frac1{T_{\text{true}}}\sum_{t}\sum_{s} w_s\,\max_{\hat q\in\hat{\mathcal Q}_s}\cos\!\big(q^{\text{true}}_{t,h},\hat q\big).$$
- **Oracle score** $S^\star$ — the perturbation score computed under $\mathcal Q_{\text{true}}$: the importance an eviction method *would* choose if it could see the future. $\mathrm{Recall@}B(S,S^\star)=\frac{|\text{top}_B(S)\cap\text{top}_B(S^\star)|}{B}$.

---

## Figure 1 — Future-query alignment  (`fig1_future_query_alignment/`)
**Question.** Do confidence-weighted *multiple* lookahead trajectories approximate the queries that actually drive decoding better than prompt-window or a single trajectory?
**Formula.** $\mathrm{Cov}(\mathcal Q_{\text{true}};\cdot)$ above, evaluated for five estimators: prompt-window; single trajectory ($M{=}1$); $M$ trajectories with uniform / log-prob / LORE-KV ($\log p-\lambda_H H$) weights. Swept over $M\in\{1,2,4,8\}$.
**Plot.** Coverage (y) vs number of particles $M$ (x); one curve per estimator.
**Why it matters.** This is the **central-premise validation**. The whole method rests on the claim that prompt-side queries are the wrong reference and that a weighted Monte-Carlo set of futures is a better one. The figure *measures* that directly instead of assuming it — and shows the entropy-weighted multi-trajectory curve dominating, with gains saturating at small $M$ (a criterion effect, not a compute effect).

## Figure 2 — Raw score comparison  (`fig2_score_comparison/`)
**Question.** How much does the importance ordering actually change from attention → single-lookahead → LORE-KV multi?
**Formula.** Three per-token scores: attention $\bar a_i=\frac1{|\mathcal Q_W|}\sum_{q\in\mathcal Q_W}a_{q,i}$; single-lookahead $S_i^{W}+\lambda_L\,\rho(E_{i,1})$; LORE-KV multi $S_i$ (Eq. 13). Pairwise Spearman $\rho$ and top-$B$ overlap.
**Plot.** Sorted score profiles (representative layer) + bars of Spearman / top-$B$ overlap between each pair.
**Why it matters.** Guards against the "it's just re-weighted attention" objection. If the three rankings were near-identical the added machinery would be pointless; the low pairwise overlap shows the lookahead objective **selects different tokens**, so any accuracy change is attributable to the criterion.

## Figure 3 — Full-cache vs evicted-cache output distribution  (`fig3_cache_distribution/`)
**Question.** Does the LORE-KV-evicted cache reproduce what the model computes at decode time better than attention-only eviction, at equal budget?
**Formula.** For each true query $q$, $o_q=\sum_j a_{q,j}v_j$ on the full cache vs on the retained set $\mathcal R$; report $\cos(o_q^{\mathcal R},o_q^{\text{full}})$ and relative $L_2$ error, for $\mathcal R$ from LORE-KV vs from attention scores.
**Plot.** Distribution (histogram) of the output cosine, and its mean across layers.
**Why it matters.** Connects the eviction *criterion* to the quantity the paper claims to preserve — the residual-stream output (Appendix A's behavioral objective). Cosine mass near 1 for LORE-KV and lower for attention-only is direct evidence the objective is the right one.

## Figure 4 — Why test-time compute  (`fig4_entropy_trajectory/`)
**Question.** Why sample *several* trajectories and down-weight uncertain ones instead of trusting one draft?
**Formula.** Per particle: entropy $\bar H_m$ vs single-trajectory alignment $\mathrm{Cov}(\mathcal Q_{\text{true}};\{\mathcal Q^{(m)}_{\text{look}}\})$; then alignment of a single (min/mean/max over $m$) vs $M$ trajectories combined (uniform / log-prob / LORE-KV).
**Plot.** Scatter (entropy vs single-trajectory alignment) + bars comparing single-draw vs combined.
**Why it matters.** Justifies the two test-time-compute design choices at once: a single random draft is high-variance and a high-entropy draft aligns poorly (scatter's negative slope), while the entropy-aware combination lifts the mean and shrinks the spread — so the extra compute buys **reliability**, not just more compute.

## Figure 5 — Confidence-weight calibration  (`fig5_confidence_calibration/`)
**Question.** Do the log-prob/entropy weights actually identify the *useful* trajectories?
**Formula.** Per particle, oracle utility $U_m$ = alignment to $\mathcal Q_{\text{true}}$, or $\mathrm{Recall@}B(S^{(m)},S^\star)$. Relate to log-prob, $-$entropy, combined $R_m$, weight $\omega_m$. Report the **critical ablation** $R^{\log p}$ vs $R^{-H}$ vs $R^{\text{combined}}$ via Spearman$(\text{signal},U_m)$.
**Plot.** (A) calibration curve — binned predicted confidence vs observed utility; (B) risk–coverage — keep the top confident fraction, plot alignment error; (C) Spearman bars per signal.
**Why it matters.** Turns entropy weighting from an "intuitive heuristic" into a **verified** one. The paper itself concedes fluent-but-wrong particles exist; panel C shows $-$entropy adds predictive power on top of log-prob, which is exactly the justification the confidence term needs.

## Figure 6 — Query-mismatch vs improvement  (`fig6_mismatch_vs_gain/`)
**Question.** Does LORE-KV help most on the very examples where prompt and future queries differ most?
**Formula.** Per example, mismatch $M(X)=1-\mathrm{Cov}(\mathcal Q_{\text{true}};\{\mathcal Q_W\})$; gain = $\mathrm{Recall@}B(S,S^\star)-\mathrm{Recall@}B(S^W,S^\star)$ and output-fidelity gain over attention. Regression slope + Spearman.
**Plot.** Scatter of gain (y) vs mismatch (x) with a fitted line.
**Why it matters.** The strongest, most causal evidence in the set: it ties the **diagnosed problem** (query mismatch) to the **observed benefit** at the individual-example level. A positive slope is the mechanism's predicted signature and is far harder to explain away than an aggregate average.

## Figure 7 — Retained-token semantic composition  (`fig7_retained_token_composition/`)
**Question.** *What kinds* of tokens does each method keep?
**Formula.** Label each token (attention-sink / recent / answer-evidence / number / entity / punctuation / other). Per class $c$: retention rate = mean fraction of heads that keep class-$c$ tokens; enrichment $=\dfrac{P(c\mid\text{retained})}{P(c\mid\text{prompt})}$. Methods: attention, AnDPro-window, LORE-KV.
**Plot.** Grouped bars of retention rate and enrichment per class per method.
**Why it matters.** Interpretability + a concrete failure-mode check: it shows LORE-KV selectively preserves **answer/evidence** tokens rather than merely recent or high-attention ones — visual, human-readable proof that the future-query criterion keeps the *right* content.

## Figure 8 — Layer-wise / head-wise analysis  (`fig8_layer_head_analysis/`)
**Question.** Where in the network does the future-query signal live?
**Formula.** Layer energy $E_l=\sum_h\sum_i S_i^{l,h}$; per-head recall gain $G_{l,h}=\mathrm{Recall@}B(S^{\text{multi}},S^\star)-\mathrm{Recall@}B(S^{W},S^\star)$; per-head mismatch $MM_{l,h}=1-\mathrm{Cov}$ (window vs true, per head).
**Plot.** $E_l$ across depth + heatmap of $G_{l,h}$ (or $MM_{l,h}$) over layers × heads.
**Why it matters.** Explains the *structure* the scores learn and pre-empts the CompressKV-style critique that heads have distinct roles: the concentration of $G_{l,h}$ in specific heads/layers both interprets the method and motivates the adaptive head-wise budget plug-in.

## Figure 9 — Budget-equivalence / memory saving  (`fig9_budget_equivalence/`)
**Question.** How much cache does LORE-KV save at equal behavioral fidelity?
**Formula.** Fidelity $F(B)$ = mean full-cache output cosine at budget $B$. Equivalent baseline budget $B^{\text{eq}}_b(B)=\min\{B':F_b(B')\ge F_{\text{LORE}}(B)\}$; saving $=1-B/B^{\text{eq}}_b(B)$.
**Plot.** $F(B)$ curves per method; memory-saving (%) vs LORE-KV budget for each baseline.
**Why it matters.** Restates the win in the currency deployment cares about — memory at equal quality — and upgrades the paper's placeholder budget-equivalence *table* into a continuous *curve*, using a fidelity metric that isn't hostage to a single accuracy threshold.

---

### How the figures fit together
- **1, 4** establish the *input* is right (the query set approximates the future).
- **2, 3** establish the *criterion* is right (it reorders importance and preserves the output).
- **5** validates the *weighting* that combines them.
- **6** links the diagnosed mismatch to the measured gain (causal evidence).
- **7, 8** interpret *what* and *where* (content and structure).
- **9** cashes the result out as memory saving.
