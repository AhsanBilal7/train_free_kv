> **SUPERSEDED (see README → "Reliability weighting: `BETA`, `GAMMA`, `LAMBDA_D`").**
> This note describes `R_m = mean_u log p - LAMBDA_H * Hbar - LAMBDA_D * zscore(D)`,
> which has been reparametrized. Two of its claims do not survive the change:
>
> * It says the disagreement term **replaces** the entropy penalty. It does not.
>   `LAMBDA_H = 0` maps to `GAMMA = 1` — a *full unit* of entropy penalty — so the
>   `tlckv-pd-*` configs are `(1, 1, 0.3)`: disagreement **on top of** entropy.
>   Disagreement *instead of* entropy is `(1, 0, 0.3)` = `config/lorekv-typ-dis-<budget>-<selection>.conf`,
>   which the old parametrization could not express.
> * It treats `LAMBDA_H = LAMBDA_D = 0` as a "log-probability only" no-penalty control.
>   It is `(1, 1, 0)`, not a no-penalty control; `(1, 0, 0)` = `config/lorekv-typ-<budget>-<selection>.conf` is.
>
> Also note `D_m` is identically 0 for `M <= 2`, not just `M = 1` (at `M = 2`,
> `D_1 = D_2` by symmetry, so `sd(D) = 0`). Everything else here — the definition of
> `D_m`, the rotary-phase argument, the standardization, the cost accounting — is
> unchanged and still current. Kept for the record.

---

# PE → PD: Trajectory Disagreement Replaces Entropy in the Particle Confidence Score

## 1. Summary

The per-particle confidence score `R_m` (Eq. 7) used to penalize a particle's **absolute
predictive entropy**. It now penalizes how **atypical that particle's future-query
trajectory is relative to the rest of the ensemble**. Nothing else in the pipeline changed:
the same lookahead particles are generated, the same perturbation score is computed, the
same head-budget allocation and chunk selection run afterwards. Only the weights
`omega_m` that combine the particles are different.

Config token: **PE** (prob + entropy) → **PD** (prob + disagreement).

## 2. The score

Before:

```
R_m = (1/T) sum_u log p(y_u^m | X, y_<u^m)  -  LAMBDA_H * (1/T) sum_u H( p(.|X, y_<u^m) )
```

After:

```
R_m = (1/T) sum_u log p(y_u^m | X, y_<u^m)  -  LAMBDA_H * mean entropy  -  LAMBDA_D * zscore_m(D)

D_m = 1/(M-1) sum_{r != m}  1/(T|G|) sum_{u,g} [ 1 - cos( q^m_{u,g}, q^r_{u,g} ) ]

omega_m = softmax( eta * R_m )
```

* `u` indexes the `T` lookahead steps, `g` ranges over the `(layer, head)` groups `G`.
* `q` are the **post-RoPE query states** captured during the lookahead phase.
* Both weights remain live, so all three regimes are reachable from config alone:
  * `LAMBDA_H = 0.3, LAMBDA_D = 0` → the published PE behaviour (bit-for-bit),
  * `LAMBDA_H = 0, LAMBDA_D = 0.3` → PD, the current method,
  * `LAMBDA_H = LAMBDA_D = 0` → log-probability only (P).

## 3. Why entropy was dropped

1. **Double counting.** The particles are sampled from `p_theta` itself, so
   `E[log p(y_u) | y_<u] = -H(p_theta(.|y_<u))`. The average log-prob term already carries
   the low-entropy signal; subtracting entropy on top counts it twice and tilts the
   estimator toward a likelihood-tilted, mode-seeking distribution instead of `p_theta`.
2. **Wrong space.** Entropy lives in token-likelihood space. The eviction score consumes
   **query directions**, which is exactly the space `D_m` is measured in.
3. **Wrong failure mode.** A confidently-wrong particle has *low* entropy and therefore
   keeps a high entropy-based confidence. Under `D_m`, a trajectory whose queries no other
   particle reproduces is down-weighted regardless of how confident the model was.

Additional property: `D_m` is **ensemble-relative**. After standardization it sums to zero
over the `M` particles, so the softmax tilt only re-allocates mass *within* the sample from
`p_theta` rather than systematically shifting the target distribution, the way an absolute
likelihood/entropy penalty does.

## 4. How `D_m` is computed

* Each decoder layer's KV cluster already buffers, per particle, the post-RoPE query states
  produced during the lookahead phase (list of `T` tensors of shape `[1, Hq, 1, D]`). `D_m`
  reuses those buffers — **no extra forward passes, no extra model calls**.
* Per layer, the per-particle buffers are concatenated over steps and stacked over
  particles into `[M, Hq, T, D]`, L2-normalized on the head dimension, and reduced with a
  single `einsum("mhtd,rhtd->mr", q, q)` averaged over `(head, step)`. Summing over layers
  and dividing by the group count gives the pairwise cosine matrix; `1 - cos`, diagonal
  zeroed, row-summed and divided by `M-1` gives `D` of shape `[M]`.
* Both queries in every compared pair sit at the **same absolute position** `n+u`, so they
  carry the same rotary phase and the cosine is directly comparable.
* **Ordering constraint:** `D_m` must be computed *before* compression / buffer reset,
  since that is what frees the per-layer lookahead query buffers. The runner does this
  right after particle generation and before `compress()`.
* **Ragged particles:** if a particle stopped early, comparison is truncated to
  `T = min(len(buffer_m))` so the `(u, g)` pairing stays valid.
* **Degenerate cases:** `M < 2` → all zeros (disagreement is undefined for a single
  particle, which matches "confidence weighting is vacuous at `M=1`"); no layer supplying
  queries → all zeros.

### Standardization

`D` enters `R_m` as a z-score across the `M` particles (zero mean, unit std, computed with
the biased std; zero spread collapses to all-zeros → uniform weights). Centering is free
because softmax is shift-invariant, but it makes the "pure re-allocation" reading exact.
The std normalization is what makes `LAMBDA_D` **scale-free**, so one setting transfers
across prompts with different absolute disagreement levels, and it bounds the tilt: a
z-score over `M` particles cannot exceed `sqrt(M-1)`.

## 5. Configuration surface

| Parameter | Default | Meaning |
|---|:---:|---|
| `LAMBDA_D` | `0.3` | Weight of the trajectory-disagreement penalty. `0.0` disables it. |
| `LAMBDA_H` | `0.0` (was `0.3`) | Legacy absolute entropy penalty, retained for ablation. |
| `DIVERGENCE_LAYER_STRIDE` | `1` | Use every k-th decoder layer when forming the group set `G`. `1` = all layers (recommended); not part of the sweep grid. |

Plumbing: both new parameters flow config → sweep loop → predictor CLI flag → model config →
KV cluster, and are read with `getattr` guards throughout, so **configs and checkpoints that
predate the change still load and reproduce their previous behaviour exactly**. The run
output-directory name now includes `lD_<value>`, so PE and PD runs never collide. The
startup banner prints both `lambda_h` and `lambda_d`.

### Config family

Ten new `PD` config files (five variants × Mistral-7B-Instruct-v0.2 / Llama-3.1-8B-Instruct)
mirror the existing PE/P/M1 groups one-for-one — same `M`, same head-budget rule, same
selection rule — so each PD config differs from its PE twin *only* in the second term of
`R_m`:

| Variant | `M` | `BUDGET_TAU` | `LAMBDA_K/V` |
|---|:---:|:---:|:---:|
| `pd-m4-adap-topk` | 4 | 1.0 (adaptive) | 0.0 (top-K) |
| `pd-m4-unif-topk` | 4 | 0.0 (uniform) | 0.0 (top-K) |
| `pd-m4-adap-div` | 4 | 1.0 (adaptive) | 0.1 (diversity) |
| `pd-mx-unif-topk` | 2/4/8 | 0.0 (uniform) | 0.0 (top-K) |
| `pd-mx-adap-div` | 2/4/8 | 1.0 (adaptive) | 0.1 (diversity) |

The controlled comparison for the confidence term is **PD-unif-topk vs PE-unif-topk vs
P-unif-topk** (disagreement vs entropy vs neither), everything else held fixed.

## 6. Cost

`D_m` adds one `M x M` cosine reduction per layer over tensors that were already
materialized: `O(M^2 · L · H · T · D)` FLOPs with no additional memory beyond the buffers
that already existed, and no additional decoding. It is negligible next to the `M · T`
lookahead decode steps that produced those buffers. `DIVERGENCE_LAYER_STRIDE > 1` can
sub-sample layers if ever needed.

## 7. Analysis / figure pipeline

* **Shared math.** The core `D_m` computation and the standardization live in the
  monkeypatch module and are imported by both the runtime path and the analysis code, so
  the figures and the deployed method score identically by construction.
* **Capture path.** The analysis capture helper now returns `particle_divergence` and
  `particle_divergence_z` alongside log-prob/entropy/weights, and applies the full
  three-term `R_m`. `D_m` is computed there **even when `LAMBDA_D = 0`**, because the
  calibration figures need it as a diagnostic signal regardless of whether the run's `R_m`
  consumes it.
* **New figure 10 — trajectory disagreement.** Answers the two questions the replacement
  needs answered on the *same* captured particles:
  * (A/B) per-particle `D_m` and entropy vs that particle's **oracle utility** (coverage of
    the true full-cache decode queries, and recall@B against the oracle perturbation
    score). `D_m` should correlate *negatively* with utility and more strongly than entropy.
  * (C) Spearman correlation bars ranking every candidate signal against utility.
  * (D) ensemble coverage of `Q_future` under uniform / log-prob / log-prob−entropy /
    log-prob−disagreement weights, plus the per-sample win rate of the disagreement rule.
    All four consume identical particles, so **the weights are the only difference — no
    extra test-time compute enters the comparison**. Paired per-sample coverages are kept
    because absolute coverage varies far more across prompts than across weightings.
  * Run with `M = 8` (`D_m` needs `M >= 2`); the diagnostic forces both `LAMBDA_H` and
    `LAMBDA_D` non-zero even if the run's config switched one off, so the two weighting
    rules stay comparable.
* **Figure 1 (future-query alignment).** Adds a `log-prob − disagreement` strategy curve
  next to the existing `log-prob − entropy` one. Because `D_m` is ensemble-relative, it is
  **recomputed within each nested prefix** of the `M` sweep rather than sliced from the
  full-ensemble value.
* **Figure 5 (confidence calibration).** Records `D_m` and its z-score per particle and adds
  `-D_hat_m` to the signal-correlation ablation.
* **Wiring.** Figure 10 is registered in the figure drivers and included in the default
  figure list; `LAMBDA_D` is threaded through the generated per-run figure config. Plot
  code skips strategies missing from older `.npz` files, so previously computed data still
  plots.

## 8. Unchanged

Particle generation (`M`, `T`, temperature, sampling), the causal perturbation/eviction
score, robust aggregation (`ROBUST_ALPHA`, `ROBUST_TOPK_BETA`), head-wise budget allocation
(`BUDGET_TAU`), diversity-aware chunk selection (`LAMBDA_K`, `LAMBDA_V`), and the
`LAMBDA_LOOKAHEAD` mix between window and lookahead scores.

## 9. Incidental fixes bundled with this change

Several figure compute scripts shadowed the per-sample checkpoint key `key` with a CUDA key
tensor from the KV cluster, which caused the resume set to store tensors and never skip an
already-finished sample. Those locals were renamed to `k_rep`/`v_rep`, restoring resume.
