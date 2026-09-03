# LORE-KV — Review, Code Audit, and Road to an Award-Quality Paper

Scope: `paper_files/main_v2 (8).tex` (= `kv_wip (44).pdf`), the implementation in
`causal_lookahead_snapkv/`, and the run harness `scripts/run_h100_aces.sh`.

---

## 0. Verdict in one paragraph

The **idea is award-shaped**: "stop asking which token *looked* important while reading,
ask which token *will matter* while answering — and treat the future as something to
*sample*, not guess." The mechanism analysis (query coverage, mismatch-vs-gain,
behavioral fidelity, budget equivalence) is already stronger than most KV-eviction
papers. But the paper is **not award-quality yet**, for three reasons that are all
fixable:

1. **The novel component doesn't earn its keep.** Everything that is inherited
  (response-side lookahead = Lookahead Q-Cache; projected deletion score = AnDPro) is
   doing the work. The paper's own numbers say so: `M=1 → 39.81`, `M=4 → 40.28`,
   `M=8 → 40.69` at B=128, while the entire reliability-weighting apparatus moves
   LongBench by **0.01** and RULER by **0.04**. A reviewer will read that as
   "ensembling + a knob that does nothing."
2. **The cost story is broken and, worse, the paper's cost claim contradicts the code.**
  2.8× latency for +1.3 points is a bad trade as presented, *and* the overhead is an
   implementation artifact, not an inherent cost (§3.1).
3. **Half the claims are** `\tbd`**.** ~40% of Table 4, all of `\lorekvD` end-to-end, and the
  key correlations in Fig. 10 are literally `--` in the submitted draft.

Sections 3–6 below say exactly what to fix, what to run, and what to add.

---



## 1. What the method actually is (my reading)

Three stages, all training-free, all at prompt time:


| Stage                 | Equation | What it does                                                                                                                                                                                     |
| --------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1. Sample the future  | Eq. 1–2  | Draw `M` continuations of length `T` from the frozen model; keep their post-RoPE query states as a Monte-Carlo sample of `Q_future`. Union with the prompt-end window `Q_W`.                     |
| 2. Weight the samples | Eq. 3–4  | `R_m = β·c̄_m − γ·H̄_m − λ_D·D̂_m`, `ω = softmax(R)`. `c_u = log q + H` is centered so `E[c_u                                                                                                    |
| 3. Score and evict    | Eq. 5–8  | Leave-one-out attention-output perturbation, projected through `W_O`: `e = ‖W_O Δo‖²`, with `Δo ≈ a/(1−a)·(v_i − o_q)`. Final score `S_i = ρ(E_{i,W}) + λ_L Σ_m ω_m ρ(E_{i,m})`. Top-B per head. |


Two shipped variants, differing only in coefficients: **/E** = `(1, 1.3, 0)` (entropy),
**/D** = `(1, 1, 0.3)` (adds disagreement). Cost: one-time `M×T` decode before the real
generation; steady-state memory and per-token decode identical to every baseline.

**The genuinely new part is stage 2 plus "M > 1".** Stage 1 is Lookahead Q-Cache at
`M=1`; stage 3 is AnDPro at `λ_L=0`. The paper says this honestly (App. B.2), which is
good practice — but honesty about a small delta is not a substitute for making the delta
matter.

---



## 2. Prior work and the claimed gap


| Line of work           | Representatives                                                                       | Limitation the paper targets                                                         |
| ---------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Prompt-side eviction   | StreamingLLM, H2O, Scissorhands, SnapKV, PyramidKV, Ada-KV, HeadKV, CAKE, DefensiveKV | Score tokens with prompt queries; the cache is later consumed by *decoding* queries. |
| Better query reference | Lookahead Q-Cache, Expected Attention, DapQ                                           | One surrogate / analytic estimate = one possible future.                             |
| Learned future utility | LookaheadKV, KVP, IndexMem                                                            | Needs offline training and model-specific parameters.                                |
| Better deletion score  | CriticalKV, AnDPro, LaProx                                                            | Improves *what* is measured, still measured at prompt-side queries.                  |


**Stated gap:** nobody combines *multi-trajectory* future-query estimation with
*future-aware output* deletion scoring, training-free. That framing is correct and
defensible. The paper's weakness is not the gap — it's that the evidence currently
supports "response-side queries help" (already known from Lookahead Q-Cache) more
strongly than it supports "*sampling* the future helps."

---



## 3. Code audit — what the implementation actually does vs. what the paper says

These are ordered by how much damage they do to the paper.

### 3.1 The particles are generated **sequentially**, not batched — the overhead is self-inflicted

- **Paper (§4.4, App. B.3):** "the `M` particles are decoded as a single batched
continuation of length `T` using the prefetched prompt KV cache."
- `config/default.conf`**:** "All M particles are decoded in parallel as one batched
continuation, so wall-clock cost equals one length-T decode, independent of M."
- **Code (**`lookahead_runner.py:101-103`**):** a Python `for m in range(M)` loop, each
iteration doing a **full deep copy of the entire prompt KV cache**
(`_clone_dynamic_cache`) and then `T` batch-size-1 forward passes.

So the actual cost is `M×T` sequential decode steps **plus** `M` **full-cache clones**
(for an 18K-token Mistral prompt that is ~4.5 GB of pure memcpy per particle), not
`1×T` batched steps. This is the direct cause of the 1.46×–2.77× numbers in Table 9 —
and of the fact that the overhead is roughly *constant in B* but large.

**Consequence:** the paper's single biggest weakness (latency) is an engineering bug,
not a property of the method. Fixing it is the highest-leverage single change available.

**Fix:** expand the prompt cache once along the batch dimension (`M`), run one batched
`T`-step decode, and slice per-particle statistics out of the `[M, ...]` tensors. All
downstream code (`step_stats`, `reliability_weights`, `divergence_from_query_stacks`)
is already written for a leading `M` axis, so this is a change local to
`lookahead_runner.py` plus the particle-index bookkeeping in the attention hijack.

### 3.2 The head-budget plug-in (`τ`) is provably a **no-op**

`snapkv_utils.py:529-536`:

```python
normalized = chunk_score / (mean_per_head.unsqueeze(-1) + eps)
E = torch.where(drop, normalized.clamp_min(0), zeros).sum(-1)   # [H]
weight = (E + eps).pow(self.budget_tau)
```

`mean_per_head` is the mean of `chunk_score` **over exactly the droppable chunks**.
Therefore `Σ_{droppable} chunk_score/mean = (Σ chunk_score)/mean = #droppable`, which is
the *same integer for every head* (the `drop` mask is head-independent). Verified
numerically: `E = [41, 41, …, 41]`, and the allocation is `1/H` for every head at both
`τ=0` and `τ=1`.

This is why Table 4 reports `+ Adaptive head budgets: 40.28` — **bit-identical** to the
core row's `40.28`. The paper currently interprets that as "the core score already
captures the gain (Fig. 8's head heterogeneity notwithstanding)". It is not evidence for
anything; the switch is disconnected. Paper Eq. 23 says `E_{l,h} = Σ_i S_i^{l,h}` (raw
scores) — implementing *that* would make `τ` real.

**Also note:** most of the paper's headline `\lorekvE` numbers were produced by
`lorekv-ent-adap-topk` configs (`adap` = `τ=1`), per `analysis/results_report_long.csv`.
Because `τ` is inert they coincide with the `unif` core, so no reported number is
*wrong* — but the provenance should be stated, or re-run under `unif-topk`.

### 3.3 The deployed score is `‖W_O Δo‖`, not `‖W_O Δo‖²`

`snapkv_utils.py:507-509` computes `dist = sqrt(dist_sq)` and scores
`|c| · ‖W_O v_i − W_O o_q‖` — the **unsquared** norm. The paper's Eq. 10 (the deployed
method) is squared; the unsquared form is introduced in App. F.1 as **analysis-only
deviation D1**, with the justification that it "preserves single-query rankings."

That justification does not hold here: the score is **averaged over** `w + MT` **queries**
before top-K, and `mean(x) ≠ mean(x²)` in rank order. So either the equation or the code
is wrong, and the difference is a real (if probably small) behavioral difference.

**Fix:** decide which one is the method, make paper and code agree, and report the other
as an ablation row ("squared vs. unsquared perturbation norm") — that row is cheap and
it is currently `\tbd` in Table 4 anyway ("Unprojected norm").

### 3.4 The protected set `𝓑` (chat-template / structural tokens) is **not implemented**

§2.1 and App. B.1 define `𝓟 = {1} ∪ 𝓦 ∪ 𝓑` with `𝓑` = chat delimiters, tool markers,
document separators, and the Ethics Statement leans on it. In `_select_chunks`
(`snapkv_utils.py:521-524`) the only protection is chunk 0 and the last `w/chunk`
chunks. There is no template-token scan anywhere in the package (`grep` for
`INST|template|delimiter` returns only prompt-*construction* code in `pred.py`).

**Fix:** either implement it (cheap, and it is a nice safety story — see §5.C.4) or
delete the claim. Right now the Ethics Statement promises a safeguard the code lacks.

### 3.5 Particles are sampled at `T=1.0` but the answer is decoded **greedily**

`lookahead_runner.py:206` — final decoding is `argmax`. Lookahead sampling is
temperature 1.0, untruncated (App. E). So the Monte-Carlo estimator targets
`p_θ(·|X)`, but the *actual* `Q_future` the cache will serve is the single **greedy**
path. The estimator is sampling from the wrong distribution.

This is not a bug so much as an **unexploited theoretical opening** — see §5.C.1. It
also explains, post-hoc, why `γ > 0` (an entropy penalty, i.e. a tilt toward
low-uncertainty / mode-seeking trajectories) helps at all: `γ` is a crude proxy for
"decode nearer the greedy path." The paper half-says this ("matching the near-greedy
deployment operating point") but never tests it.

### 3.6 Smaller items

- **Seeds are hard-wired.** `pred.py:281` calls `seed_everything(42)`; there is no
`--seed` flag and `run.sh` never passes one. The three-seed plan written into
`run_h100_aces.sh` (`SEED=$seed PYTHONHASHSEED=$seed`) is **silently ignored** — those
runs would be three identical runs. Given the paper explicitly says the gaps at stake
(0.01, 0.04) may be below the noise floor, this must be plumbed through before any
reliability conclusion is drawn.
- **The latency table is not apples-to-apples.** Baselines go through HF
`model.generate` (`pred.py:216, 226`); LORE-KV goes through a hand-written per-token
Python loop (`lookahead_runner.py:198-211`). Part of the reported 2.8× is decode-loop
overhead unrelated to lookahead. Report **prefill-side overhead separately from
per-token decode**, or run both through the same loop.
- **Head scoring is a Python loop over** `H` (`snapkv_utils.py:480`) — 32 heads × 32
layers of small matmuls. Batching it with `einsum` is another easy multiplier on the
overhead story.
- **Baseline coverage is thin.** Only `andpro` and `critical` are actually re-run;
SnapKV/PyramidKV/Ada-KV/H2O numbers in Table 1 are copied from AnDPro. Fine for
Mistral-LongBench (matched protocol), risky elsewhere.
- **No Lookahead Q-Cache baseline end-to-end.** It is approximated by "our method at
`M=1`". That is a *favorable* approximation (same score, same code, same protected
set) and reviewers will notice the closest competitor is absent from every table.

---



## 4. Why it is not award-quality yet — the reviewer's-eye summary


| #   | Objection a reviewer will raise                                                 | Current state                                                                                       |
| --- | ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| R1  | "The novelty is the reliability weighting, and it does nothing (0.01 / 0.04)."  | Unanswered. `\lorekvD` end-to-end is `\tbd` everywhere.                                             |
| R2  | "You spend 2.8× compute for +1.3 points. Give the baseline the same compute."   | Control (iv) is "specified but not yet run".                                                        |
| R3  | "`M` is non-monotone (`M=2` < `M=1`). What is your noise floor?"                | Unknown; single seed; no error bars anywhere.                                                       |
| R4  | "Two 7–8B models, ≤18K context. Does it hold at 70B / 128K / GQA-heavy models?" | Not tested.                                                                                         |
| R5  | "40% of your ablation table is `--`."                                           | True as submitted.                                                                                  |
| R6  | "This is Lookahead Q-Cache + AnDPro. What is the intellectual contribution?"    | The centering/reparametrization (Prop. 1) is genuinely nice but is currently framed as bookkeeping. |
| R7  | "The paper has `\todom{}` margin notes and placeholder authors."                | Cosmetic, but must be gone.                                                                         |


**The strategic read:** you cannot win R1 by finding a bigger number for `λ_D`. You win
by **reframing the contribution around what the evidence actually supports** and adding
the two or three experiments that make that reframing undeniable. See §5.

---



## 5. What to add — ranked by impact per GPU-hour



### A. P0 — Must-fix before submission (credibility)


| A#  | Action                                                                                                                                          | Why                                                                                                                                                                                                                                                 |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A1  | **Batch the** `M` **particles** (§3.1).                                                                                                         | Turns the paper's worst number into one of its best. Expected: overhead drops from `M×T` to `~1×T` sequential steps, i.e. Table 9's 2.77× should fall toward ~1.2–1.4×. This single fix likely changes reviewer sentiment more than any new result. |
| A2  | **Plumb** `--seed` **and run 3 seeds** at `B=128` on LongBench + RULER for `{M=1, M=2, M=4, M=8}` and the reliability cells. Report mean ± s.d. | Directly answers R3. Without it, every sub-0.5-point claim in the paper is unsupported. Also lets you say honestly "M=2 < M=1 is within noise", which *strengthens* the saturation claim (P2).                                                      |
| A3  | **Fix or delete** `τ` (§3.2).                                                                                                                   | An ablation row that is bit-identical to the core row is a red flag any careful reviewer will catch.                                                                                                                                                |
| A4  | **Reconcile Eq. 10 with the code** (§3.3) and add the squared-vs-unsquared row.                                                                 | Reproducibility. Currently the stated method is not the run method.                                                                                                                                                                                 |
| A5  | **Fill every** `\tbd`**, or cut the claim.**                                                                                                    | Especially: `\lorekvD` end-to-end at `B=128`, the `(0,0,0)` uniform control, the `(1,0,0)` no-entropy control, and the Fig. 10 correlations.                                                                                                        |
| A6  | **Run the compute-matched control** (App. B.3-(iv)): feed the *identical* `M` continuations to SnapKV-style and AnDPro-style scoring.           | This is R2's answer and it is already designed. It is the difference between "we bought accuracy with compute" and "we bought accuracy with a better criterion." **Highest scientific value of anything on this list.**                             |
| A7  | **Add Lookahead Q-Cache as a real end-to-end baseline**, not as "ours at `M=1`".                                                                | The closest competitor must appear in Table 1.                                                                                                                                                                                                      |
| A8  | Strip `\todom{}`, `\todo`, `\textcolor{red/blue}`, placeholder authors.                                                                         | Obvious.                                                                                                                                                                                                                                            |




### B. P1 — Experiments that turn this into a *mechanism* paper

These are the ones that make it award-quality rather than merely publishable. The theme:
**stop arguing that LORE-KV wins; argue that you now understand *when* future-query
estimation is worth anything at all.** That is a contribution nobody else has made, and
it survives even if the numbers stay modest.


| B#     | Experiment                                                                                                                                                                                                                                                                                                                                                  | Reasoning                                                                                                                                                                                                                                                                                                                                                                                                 |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **B1** | **The adaptive-lookahead policy.** Your own conclusion proposes it and then doesn't run it: predict `Mis(X)` cheaply at prompt time (e.g. from the spread of window queries, or from one 1-token probe), and enable lookahead *only* above a threshold. Report accuracy vs. average overhead as a Pareto curve against the always-on and never-on extremes. | This converts the paper's central *finding* (gain ∝ mismatch, ρ≈0.5) into a *method*. It answers R2 in the strongest possible way — "we don't pay the cost when it wouldn't help" — and gives a headline like *"90% of the gain at 30% of the overhead."* This is the single best new-experiment candidate.                                                                                               |
| **B2** | `T` **sweep** (`T ∈ {2, 4, 8, 16, 32}`) at fixed `M`, and the joint `(M, T)` grid at *fixed total decode budget* `M·T = const`.                                                                                                                                                                                                                             | App. C says "varying `T` is a planned axis we have not yet run." A reviewer will ask "why 8?" and, more sharply, "at fixed compute, is it better to sample 8 futures shallowly or 2 deeply?" That is a genuinely interesting question about future-query estimation and nobody has answered it.                                                                                                           |
| **B3** | **Lookahead-temperature sweep** (`τ_look ∈ {0 (greedy), 0.3, 0.7, 1.0, 1.3}`) with query coverage of the *true* greedy `Q_future` as the measured quantity.                                                                                                                                                                                                 | Directly probes §3.5's distribution mismatch. Predicted result: coverage peaks below 1.0. If so, you have (a) a free accuracy gain, (b) a principled replacement for the entropy penalty, and (c) an explanation of *why* `γ>0` helped — which retroactively rescues the reliability story.                                                                                                               |
| **B4** | **Noise-floor / error-bar table** for the whole reliability plane (`(0,0,0)`, `(1,0,0)`, `(1,1,0)`, `(1,1.3,0)`, `(1,2,0)`, `(1,0,0.3)`, `(1,1,0.3)`) at `B=128`, 3 seeds.                                                                                                                                                                                  | Turn R1 from a weakness into a *result*: **"reliability weighting is within noise; the ensemble is what matters."** A paper that decisively kills its own sub-component is more credible, not less — and it makes the `M`-sweep the real finding.                                                                                                                                                         |
| **B5** | **Scale study:** one larger backbone (Qwen2.5-14B / Llama-3.1-70B-Instruct or a strong GQA model) and one long-context point (64K–128K, RULER or ∞Bench).                                                                                                                                                                                                   | Answers R4. Long context is where the query-mismatch argument is *strongest* (your own RULER trend: gain grows 3.58 → 5.85 from 4K → 16K). Extrapolating that curve to 64K is the most persuasive single plot you could produce.                                                                                                                                                                          |
| **B6** | **Reasoning / long-generation tasks** (GSM8K-hard, MATH, or an agentic trace with 512+ output tokens).                                                                                                                                                                                                                                                      | Your entire thesis is "prompt-end queries mispredict decoding queries." That mismatch is *maximal* when the model generates a long chain of thought whose queries drift far from the prompt. LongBench outputs are short (≤512 tokens), which is the regime *least* favorable to your claim — you are currently testing your hypothesis in its weakest setting. This is the highest-upside new benchmark. |
| **B7** | **Orthogonality demo:** LORE-KV + KV quantization (KIVI / 4-bit) at matched total bytes.                                                                                                                                                                                                                                                                    | Your Related Work argues quantization is complementary; showing `evict-then-quantize` compounds converts a claim into a result and broadens impact. Cheap if a quantized-cache path exists.                                                                                                                                                                                                               |
| **B8** | **Failure-mode analysis, honestly reported.** You already know code completion regresses on Llama at `B=128` and multi-query needles regress on RULER. Characterize *why* (hypothesis: lookahead continuations for code are locally-syntactic and pull retention toward the recent window).                                                                 | Award committees reward papers that explain their own failures. A one-figure diagnosis here is worth more than another +0.2 average.                                                                                                                                                                                                                                                                      |




### C. P2 — Methodological additions that would add real novelty

Ranked by (novelty × feasibility). These are what would make the difference between "a
good empirical paper" and "a paper people cite for the idea."

**C1. Decode the particles from the *deployment* law, and say why.** (Builds on B3.)
Right now you sample from `p_θ` and then bolt on an entropy penalty to tilt back toward
the mode. Cleaner and more defensible: define the estimator's target explicitly as the
distribution of queries the deployed decoder will actually produce, and sample from a
temperature-`τ_look` law chosen to match it. This makes `γ` *derivable* rather than
tuned, replaces a weak empirical knob with a principled one, and gives you a real
theoretical contribution instead of Prop. 1's bookkeeping.

**C2. Make disagreement do something other than reweight.** `D̂_m` currently only tilts
softmax weights — a second-order effect on a first-order quantity, which is exactly why
it moves nothing. Two much stronger uses of the *same* signal, at the same cost:

- **Adaptive** `M` **(sequential stopping).** Draw particles until the ensemble's query
coverage stops moving (measurable online via `D_m`), then stop. Your Fig. 1 shows
saturation at ~`M=4` *on average* — but the variance across prompts is where the
compute savings live. This is the natural companion to B1 and turns the paper's
"gains saturate" observation into an algorithm.
- **Disagreement-aware *budget* rather than disagreement-aware *weights*.** High
ensemble disagreement = genuinely uncertain future = the case where a single top-K
retention is riskiest. Allocate more budget (or switch to robust/top-κ aggregation)
for high-`D` prompts. This makes the signal act where it has leverage.

**C3. Determinantal / coverage-based particle selection.** You sample `M` i.i.d.
continuations, many of which are near-duplicates — which is precisely why coverage
saturates at `M≈4` and why `M=2` can underperform `M=1`. Sampling for **query-space
coverage** instead (diverse beam search, DPP over the `q` trajectories, or simply
rejecting particles with `D_m` below a threshold) attacks the estimator's variance
directly. Framing: *"the bottleneck is not more samples, it is better-spread samples"* —
a genuinely new claim in this literature, and it makes `M=8`-quality coverage reachable
at `M=2` cost.

**C4. Evidence-preservation / safety guarantee under aggressive eviction.** Implement
`𝓑` (§3.4), then measure instruction-following survival under compression: multi-instruction
prompts, prompt-injection-resistance, or "did the system prompt survive at B=128?".
Your Fig. 7 already shows LORE-KV retains answer-bearing tokens at 2.5× enrichment —
extend that to *instructions*, and you have a safety story no competing eviction paper
has. The Ethics Statement already promises this analysis.

**C5. A cheap analytic surrogate for the lookahead.** The strongest possible version of
this paper's thesis: show that a *free* estimate of `Q_future` (e.g. a learned-free
linear extrapolation of the query trajectory, or Expected-Attention-style analytic
queries) recovers `X%` of the sampled-lookahead gain. Even a negative result is
valuable: it establishes that *actually sampling* is necessary, which is exactly the
claim the title makes and which nothing in the paper currently tests.

### D. P3 — Framing and presentation

1. **Retitle the contribution around the finding, not the mechanism.** The paper's most
  defensible, most quotable result is not "LORE-KV wins by 1.3 points." It is:
  > *Response-side query estimation is what matters; it saturates after ~4 samples; and
  > its benefit is predictable per-example from prompt-to-future query mismatch.*
   That is a **mechanistic law with an operational consequence** (B1's adaptive policy).
   Lead with it. The current abstract already gestures at this in the red text — promote
   it to the thesis.
2. **Answer the "why not just train it?" question head-on** (the author's own `\todom`
  at line 94). The defense is concrete and you have most of the data: (a) zero offline
   cost and no per-model artifacts — a learned predictor must be retrained per backbone,
   per tokenizer, per fine-tune; (b) it adapts *per input* at test time, whereas a
   trained predictor is frozen at its training distribution — and your Fig. 6 shows the
   gain is highly input-dependent (ρ≈0.5), i.e. exactly the regime where a fixed
   predictor generalizes worst; (c) no distribution-shift failure mode when the
   deployment domain differs from the predictor's training data. Add one experiment if
   possible: a domain (code, or a non-English task) where a learned method's training
   distribution plainly doesn't cover.
3. **Restructure Table 4** so single-factor ablations are visually separated from
  multi-switch rows. Right now "+ Robust aggregation" toggles three switches
   (`α`, `κ`, `τ`) and sits in the same block as single-factor rows; the footnote
   admits this. Reviewers skim tables, not footnotes.
4. **Merge Figures 1 and 6 into a single "mechanism" figure** on page 2–3. The story
  *(coverage improves → coverage saturates → gain tracks mismatch)* is your strongest
   asset and it currently lives in the appendix. Move it into the main body and cut a
   results table to make room.
5. **Report overhead as a fraction of end-to-end serving cost, amortized.** The one-time
  `M×T` cost is fixed while decode cost grows with output length. Plot overhead % vs.
   output length: at 512+ output tokens the lookahead is a small constant. Combined with
   A1 this reframes the cost objection entirely.

---



## 6. Concrete run plan (mapped onto `scripts/run_h100_aces.sh`)

The harness already supports everything in Tier 1. Ordered by dependency.

**Tier 0 — code fixes first (no GPU).** A1 batching, A2 `--seed` plumbing, A3 `τ`,
A4 squared-norm. All four are small, local edits. *Nothing below is worth running until
A2 lands* — single-seed numbers cannot resolve the effects you care about.

**Tier 1 — the completeness sweep (harness is ready today).**

```bash
# Reliability plane, 3 seeds, B=128, both benchmarks — kills or confirms R1/B4.
for seed in 0 1 2; do
  for cell in uniform typ logp ent ent-only; do
    EXPERIMENT=LongBench CONFIG_FILE=config/lorekv-$cell-unif-topk.conf \
      BUDGET=128 SEED=$seed MODEL_TAG="Mistral-7B-Instruct-v0.2-s$seed" \
      bash scripts/run_h100.sh
  done
done
# Then the same on RULER, then the gated Stage-2 cells (dis, typ-dis) *only if* the
# S4 semipartial test passes per-benchmark (Fig. 10 already shows rho = -0.26 on
# LongBench vs +0.03 on RULER — a pooled number would hide that sign flip).
```

Also here: the `M` sweep at 3 seeds (`NUM_PARTICLES="0 1 2 4 8"`), and the `\lorekvD`
end-to-end rows.

**Tier 2 — new axes (small harness additions).**

- B2: expose `LOOKAHEAD_LEN` in the sweep grid → `T ∈ {2,4,8,16,32}`, plus the fixed
`M·T` diagonal.
- B3: `LOOKAHEAD_TEMPERATURE ∈ {0, 0.3, 0.7, 1.0, 1.3}` (already a config key; just needs
the greedy-`τ=0` path).
- A6: a new `mode` (`snapkv_lookahead` / `andpro_lookahead`) reusing the captured
particles — this is a scoring-path change only, no new decoding.

**Tier 3 — new science.**

- B1 adaptive policy (needs a cheap online `Mis(X)` proxy — start with the score-level
proxy already computed in `fig6_mismatch_vs_gain/`).
- B6 long-generation / reasoning benchmark (new `experiments/` directory).
- B5 scale study (new backbone; check the hijack covers its attention class).
- C2/C3 (adaptive `M`, diverse particles) — build on the same capture pipeline.

---



## 7. If you only do five things

1. **A1** — batch the particles. Turns 2.8× into ~1.3× and removes the single biggest
  objection. Pure engineering, no new science needed.
2. **A6** — the compute-matched control. Without it, "better criterion, not more
  compute" is an assertion; with it, it is the paper's central proven claim.
3. **A2 + B4** — seeds and error bars across the reliability plane. Either the weighting
  works (great) or it demonstrably doesn't (also great, if you say so first).
4. **B1** — the adaptive-lookahead policy. Converts your mechanism finding into a method
  and gives the paper a memorable headline number.
5. **B6** — long-generation / reasoning evaluation. Tests your hypothesis in the regime
  where it should be *strongest*, instead of only where it is weakest.

Items 1–3 make the paper *solid*. Items 4–5 are what make it *award-quality*: they move
the contribution from "a better eviction score" to "a predictive account of when
future-query estimation matters, and an algorithm that exploits it."