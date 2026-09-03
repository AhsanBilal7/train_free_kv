# AnDPro: Accurate KV Cache Eviction via Anchor Direction Projection for Efficient LLM Inference

This repository contains the code for our paper **"Accurate KV Cache Eviction via Anchor Direction Projection for Efficient LLM Inference"**, which has been accepted by NeurIPS 2025.

Paper: [https://openreview.net/pdf?id=Tdl89SZItB](https://openreview.net/pdf?id=Tdl89SZItB)


## Usage of this Repo

### Requirements
The requirements are the same as [AdaKV](https://github.com/FFY0/AdaKV).

```
transformers==4.44.2
uv pip install flash-attn --no-build-isolation

datasets
tiktoken
jieba
rouge_score
pandas          # RULER scorer (experiments/RULER/eval.py)
```

### Installation

```
git clone git@github.com:MIRALab-USTC/LLM-AnDPro.git
cd LLM-AnDPro
make i
```

### Quick Start

Two benchmarks are supported, selected with the `EXPERIMENT` variable:

| `EXPERIMENT` | Benchmark | Dataset (`DATASET_PATH`) | Subsets (`DATASET_NAME_LIST`) |
|---|---|---|---|
| `LongBench` *(default)* | [LongBench](experiments/LongBench/) | `THUDM/LongBench` | 16 sub-tasks (qasper, hotpotqa, …) |
| `RULER` | [RULER](experiments/RULER/) | `simonjegou/ruler` | sequence-length configs: `4096 8192 16384` |

Run the full pipeline (prediction + evaluation) from the repo root, switching
benchmarks with a single variable:

```bash
# LongBench (default)
bash scripts/run_h100.sh

# RULER — just change the experiment
EXPERIMENT=RULER bash scripts/run_h100.sh
```

### Analysis / paper figures

The mechanism figures live in `experiments/<BENCHMARK>/figure_making_script/`
(each figure is a `compute_*.py` that saves a `.npz` + a `plot_*.py` that draws the
`.png`). One driver runs them all with the same scratch/HF/GPU exports as
`run_h100.sh`:

```bash
sbatch scripts/run_figures.sh                        # all 9, LongBench single-doc QA, B=128
bash   scripts/run_figures.sh                        # same, interactive
FIGURES="1 2 6" bash scripts/run_figures.sh          # subset
BENCHMARK=ruler DATASET=8192 bash scripts/run_figures.sh
BUDGET=256 SAMPLES=5 CONFIG_FILE=config/lorekv-ent-unif-topk.conf bash scripts/run_figures.sh
PLOT_ONLY=1 bash scripts/run_figures.sh              # re-draw from saved .npz (no GPU)
```

See `experiments/LongBench/figure_making_script/README.md` for what each figure shows.

**Figures inside a real run.** Add `MAKE_FIGURES=1` to any `run_h100.sh`
`causal_lookahead` run to also drop every figure's `.npz` + `.png` into that run's
own output folder (`experiments/<EXP>/pred/<out_name>/`) — one artifact **per
dataset** plus one **combined** across all datasets in the run, computed with the
run's exact hyper-parameters:

```bash
MAKE_FIGURES=1 FIG_SAMPLES=2 FIG_LIST="1 2 6" \
  CONFIG_FILE=config/lorekv-ent-adap-topk.conf bash scripts/run_h100.sh
```

This is opt-in (default off) and heavy — each (figure × dataset) reloads the model,
so scope it with `FIG_SAMPLES`, `FIG_LIST`, and the run's `DATASET_NAME_LIST`. The
standalone `scripts/run_figures.sh` (writes into the `figure_making_script/` folders
instead) remains available for iterating on figures outside a run.

The dataset and its subset list are defined in each benchmark's own `run.sh`
(`experiments/<EXPERIMENT>/run.sh`), so switching benchmarks only requires changing
`EXPERIMENT`. You can still run a benchmark directly (set the GPU yourself):

```bash
cd experiments/RULER   # or experiments/LongBench
bash run.sh
```

### SLURM Submission (H100)

Use [scripts/submit.sh](scripts/submit.sh) to submit any named config as a SLURM job. It automatically sets the job name and config file — just pass the config name (without the path or `.conf` extension):

```bash
bash scripts/submit.sh <config-name>
```

This defaults to LongBench. To run the **same config on RULER**, set `EXPERIMENT` — it is forwarded to the job via `--export=ALL`:

```bash
EXPERIMENT=RULER bash scripts/submit.sh <config-name>
```

#### All configs — copy-paste commands

> **Current family first.** The `lorekv-*` configs below are the reparametrized
> `(BETA, GAMMA, LAMBDA_D)` grid and are what new experiments should use; see
> [Running the reparametrized experiments](#running-the-reparametrized-experiments).
> The `tlckv-*` groups after them are the legacy family, kept reproducible — a config
> that sets `LAMBDA_H`/`PARTICLE_ETA` is auto-mapped to `(BETA, GAMMA, LAMBDA_D)` by
> `run_h100.sh`, which prints the mapping to the log.

Configs are named `lorekv-<cell>-<budget>-<selection>[-llama]`; `unif-topk` is the
core configuration. See [The config matrix](#the-config-matrix).

```bash
# Reliability sweep — Stage 1, the (BETA, GAMMA) plane at LAMBDA_D = 0.
# Budget/selection held at the core unif-topk, so this varies ONE axis.
BUDGET=128 bash scripts/submit.sh lorekv-uniform-unif-topk   # (0, 0, 0)   uniform control
BUDGET=128 bash scripts/submit.sh lorekv-typ-unif-topk       # (1, 0, 0)   no-entropy control
BUDGET=128 bash scripts/submit.sh lorekv-logp-unif-topk      # (1, 1, 0)   = old "log-prob only"
BUDGET=128 bash scripts/submit.sh lorekv-ent-unif-topk       # (1, 1.3, 0) = old LORE-KV /E
BUDGET=128 bash scripts/submit.sh lorekv-ent-only-unif-topk  # (0, 1, 0)   only if D1 warrants it

# Stage 2 — the LAMBDA_D probe, gated on test S4
BUDGET=128 bash scripts/submit.sh lorekv-dis-unif-topk       # (1, 1, 0.3) = old LORE-KV /D
BUDGET=128 bash scripts/submit.sh lorekv-typ-dis-unif-topk   # (1, 0, 0.3) disagreement, no entropy

# Allocation/selection sweep — reliability held fixed, the other two axes vary
BUDGET=128 bash scripts/submit.sh lorekv-ent-unif-topk
BUDGET=128 bash scripts/submit.sh lorekv-ent-adap-topk
BUDGET=128 bash scripts/submit.sh lorekv-ent-unif-div
BUDGET=128 bash scripts/submit.sh lorekv-ent-adap-div

# Lookahead-budget sweep — no dedicated configs; NUM_PARTICLES is an override
NUM_PARTICLES=1       BUDGET=128 bash scripts/submit.sh lorekv-ent-unif-topk
NUM_PARTICLES="2 4 8" BUDGET=128 bash scripts/submit.sh lorekv-ent-unif-topk

# RULER / Llama: prepend EXPERIMENT=RULER, or append -llama to the config name
EXPERIMENT=RULER BUDGET=128 bash scripts/submit.sh lorekv-ent-unif-topk
BUDGET=128 bash scripts/submit.sh lorekv-ent-unif-topk-llama
```

### For LongBench (legacy family)
```bash
# Group A — Adaptive budget, Top-K selection
bash scripts/submit.sh lorekv-ent-adap-topk
bash scripts/submit.sh lorekv-logp-adap-topk
bash scripts/submit.sh lorekv-ent-adap-topk

# Group B — Uniform budget, Top-K selection
bash scripts/submit.sh lorekv-ent-unif-topk
bash scripts/submit.sh lorekv-logp-unif-topk
bash scripts/submit.sh lorekv-ent-unif-topk

# Group C — Adaptive budget, Diversity-aware selection
bash scripts/submit.sh lorekv-ent-adap-div
bash scripts/submit.sh lorekv-logp-adap-div
bash scripts/submit.sh lorekv-ent-adap-div
```

### For RULER (legacy family)
```bash
# Group A — Adaptive budget, Top-K selection
EXPERIMENT=RULER bash scripts/submit.sh lorekv-ent-adap-topk
EXPERIMENT=RULER bash scripts/submit.sh lorekv-logp-adap-topk
EXPERIMENT=RULER bash scripts/submit.sh lorekv-ent-adap-topk

# Group B — Uniform budget, Top-K selection
EXPERIMENT=RULER bash scripts/submit.sh lorekv-ent-unif-topk
EXPERIMENT=RULER bash scripts/submit.sh lorekv-logp-unif-topk
EXPERIMENT=RULER bash scripts/submit.sh lorekv-ent-unif-topk

# Group C — Adaptive budget, Diversity-aware selection
EXPERIMENT=RULER bash scripts/submit.sh lorekv-ent-adap-div
EXPERIMENT=RULER bash scripts/submit.sh lorekv-logp-adap-div
EXPERIMENT=RULER bash scripts/submit.sh lorekv-ent-adap-div
```

Each call expands to the equivalent of:
```bash
sbatch --job-name=<EXPERIMENT>-<config-name> \
    --export=ALL,EXPERIMENT=<EXPERIMENT>,CONFIG_FILE=config/<config-name>.conf scripts/run_h100.sh
```
(`EXPERIMENT` defaults to `LongBench`; the job name is prefixed with it so LongBench and RULER runs of the same config are easy to tell apart.)

Logs are written to `/home/ahsan/LLM-AnDPro/logs/out_andpro_h100_<jobid>.log` and `err_andpro_h100_<jobid>.log`.

---

## TLC-KV Configuration Reference

`causal_lookahead` implements the full TLC-KV method (Algorithm 1). The named configurations below isolate specific design choices as controlled ablations. Every config uses `MODE=causal_lookahead` with **plain mean score aggregation** (`ROBUST_ALPHA=0.0`, `ROBUST_TOPK_BETA=1.0`) fixed throughout — the three axes that vary are the **particle confidence signal**, the **head-wise budget allocation**, and the **chunk selection rule**.

Each config has a matching `.conf` file in [`config/`](config/) that can be passed directly. These files set only the **method** hyperparameters (mode, budget, particles, aggregation, selection) and are **dataset-agnostic** — the dataset lives in each benchmark's `run.sh` — so the same config runs on either benchmark. Prepend `EXPERIMENT=RULER` to evaluate any of them on RULER:
```bash
CONFIG_FILE=config/<name>.conf bash scripts/run_h100.sh                    # LongBench (default)
EXPERIMENT=RULER CONFIG_FILE=config/<name>.conf bash scripts/run_h100.sh   # RULER
```

> The inline `... bash scripts/run_h100.sh` commands in each config block below run on LongBench by default; prepend `EXPERIMENT=RULER` (and optionally `DATASET_NAME_LIST="4096 8192 16384"`) to run them on RULER instead.

> **`LAMBDA_H` / `PARTICLE_ETA` in the Group A/B/C blocks are legacy names.** They still
> work — `run_h100.sh` maps them to `(BETA, GAMMA, LAMBDA_D) = (eta, eta*(1+LAMBDA_H),
> eta*LAMBDA_D)` and logs the result — but new commands should set `BETA`/`GAMMA`
> directly. Passing both means the explicit `BETA`/`GAMMA` win and the legacy values are
> ignored (with a warning).

### Naming key

| Token | Parameter(s) | Value | Meaning |
|---|---|:---:|---|
| `<cell>` | `BETA`,`GAMMA`,`LAMBDA_D` | — | the reliability weighting; one of `uniform` `typ` `logp` `ent` `ent-only` `dis` `typ-dis`. See [Reliability weighting](#reliability-weighting-beta-gamma-lambda_d). |
| **Adap** | `BUDGET_TAU` | 1.0 | **Adaptive** head-wise budget: each head's KV budget is proportional to its total future-query perturbation mass. Heads where deletions matter most get more tokens. |
| **Unif** | `BUDGET_TAU` | 0.0 | **Uniform** budget: every head receives an equal share of the layer budget, disabling adaptive allocation. This is the core method's setting. |
| **TopK** | `LAMBDA_K`, `LAMBDA_V` | 0.0 | **Plain top-K** selection: candidates ranked by score alone, no diversity penalty. |
| **Div** | `LAMBDA_K`, `LAMBDA_V` | 0.1 | **Diversity-aware** greedy selection: penalises candidates whose keys/values are already well represented by earlier picks. Useful for tasks with near-duplicate passages; keep values small so it only breaks ties. |

The full matrix is in [The config matrix](#the-config-matrix) below.

All configs also share: `NUM_PARTICLES=4` (except M1), `LOOKAHEAD_LEN=8`, `LAMBDA_LOOKAHEAD=1.0`, `LOOKAHEAD_TEMPERATURE=1.0`, `ROBUST_ALPHA=0.0`, `ROBUST_TOPK_BETA=1.0`.

<a id="reliability-weighting-beta-gamma-lambda_d"></a>
### Reliability weighting: `BETA`, `GAMMA`, `LAMBDA_D`

**The problem the old score had.** The particles are drawn from `p_theta` itself, so
for every sampled token

```
E[ log p(y_u | h_u) | h_u ]  =  -H( p(.|h_u) ).
```

The log-probability of a sampled token **already contains** the entropy of the
distribution it was drawn from. Subtracting `LAMBDA_H * mean_u H_u` on top of it did
not add a second signal — it rescaled the one already present. That is why
`LAMBDA_H = 0` vs `0.3` moved LongBench by 0.01 points.

**The fix.** Add the entropy back at the level of a single step, giving the *centered*
score

```
c_u = log q(y_u | h_u) + H( q(.|h_u) ),        E[c_u | h_u] = 0
```

which measures **only** how improbable the branch actually taken was, *relative to the
uncertainty at that context*. A trajectory is then scored with three independent
coefficients:

```
R_m      = BETA * cbar_m  -  GAMMA * Hbar_m  -  LAMBDA_D * zscore_m(D)
omega_m  = softmax(R)_m
cbar_m   = (1/T) sum_u c_u        Hbar_m = (1/T) sum_u H_u
```

| Coefficient | Prices | Notes |
|---|---|---|
| `BETA` | **branch atypicality** — did the particle take steps the model itself deemed improbable at the contexts it visited? | `E[cbar_m] = 0`, so this channel contributes no zeroth-order shift to the estimator's target. |
| `GAMMA` | **context uncertainty** — an explicit, deliberate preference for low-entropy histories, matching the near-greedy operating point the deployed system decodes from. | `GAMMA = 0` turns the entropy channel off entirely. |
| `LAMBDA_D` | **ensemble disagreement** — how far the particle's future-query trajectory sits from the rest of the ensemble. | Off by default. Identically 0 for `M <= 2`. |

There is **no `PARTICLE_ETA`**: `softmax(eta*(b*c - g*H - lD*D))` equals
`softmax((eta*b)*c - (eta*g)*H - (eta*lD)*D)`, so four symbols carried three degrees of
freedom. It has been absorbed into the three coefficients.

#### Nesting: reproducing any older run

Because `(1/T) sum_u log q(y_u|h_u) = cbar_m - Hbar_m`, the old score is exactly

```
R_old = cbar_m - (1 + LAMBDA_H) * Hbar_m - LAMBDA_D * Dhat_m
```

so `(BETA, GAMMA, LAMBDA_D) = (eta, eta*(1 + LAMBDA_H_old), eta*LAMBDA_D_old)`
reproduces any published run bit-for-bit:

| Old configuration | `(BETA, GAMMA, LAMBDA_D)` | Config file |
|---|:---:|---|
| uniform weights | `(0, 0, 0)` | `lorekv-uniform-unif-topk` |
| "log-probability only" (`LAMBDA_H=LAMBDA_D=0`) | `(1, **1**, 0)` | `lorekv-logp-unif-topk` |
| LORE-KV /E (`LAMBDA_H=0.3`) | `(1, 1.3, 0)` | `lorekv-ent-unif-topk` |
| LORE-KV /D (`LAMBDA_D=0.3`) | `(1, **1**, 0.3)` | `lorekv-dis-unif-topk` |
| — *(unreachable before)* pure typicality | `(1, 0, 0)` | `lorekv-typ-unif-topk` |
| — *(unreachable before)* disagreement **instead of** entropy | `(1, 0, 0.3)` | `lorekv-typ-dis-unif-topk` |

**`scripts/run_h100.sh` applies this map automatically**: a legacy `config/tlckv-*.conf`
that sets `LAMBDA_H` / `PARTICLE_ETA` and no `BETA`/`GAMMA` is translated on the fly and
the mapping is printed to the log, so every existing config keeps working and keeps
producing the same numbers.

Two things this makes visible, both of which were misread before:

* `LAMBDA_H = LAMBDA_D = 0` is `GAMMA = 1` — a **full unit** of entropy penalty. The
  `tlckv-p-*` "log-probability only" configs are *not* a no-penalty control.
* `PE` and `PD` do **not** "differ only in which penalty is active": `(1, 1.3, 0)` vs
  `(1, 1, 0.3)` differ in **both** `GAMMA` and `LAMBDA_D`. The single-factor test of the
  disagreement hypothesis is `lorekv-typ-dis-*` = `(1, 0, 0.3)` against `lorekv-typ-*` = `(1, 0, 0)`.

#### `NORMALIZE_TYPICALITY` (ablated switch, default `0`)

The `c_u` are a martingale difference sequence, so `Var(cbar_m) = (1/T^2) sum_u E[V_u]`
where `V_u = Var_{y~q}[log q(y)]` is the varentropy. The across-particle spread of
`cbar_m` therefore scales with the prompt's varentropy, and one global softmax scale is
not calibrated across prompts. Setting `NORMALIZE_TYPICALITY=1` uses
`ctilde_m = cbar_m / sqrt(Vbar_m / T)` instead, which has unit scale by construction.
Report it as a switch, not a default — test **V5** below decides whether it earns a place.

#### Sampling law == centering law

`E[c_u | h_u] = 0` holds **only** when the particles are drawn from the same law `c_u`
is centered against. We sample at `LOOKAHEAD_TEMPERATURE=1.0` with no truncation, so
`q = p_theta` and the identity is exact. If you add top-k / top-p / a different
temperature, the statistics must be computed from the **same shaped logits the sampler
consumed** — which is what `snapkv_utils.step_stats()` does. Centering a truncated
sample against the untruncated `p_theta` makes `E[c_u] > 0`: a silent positive bias with
no other visible symptom. Test **V1** catches it.

---

### The config matrix

Every config is `lorekv-<cell>-<budget>-<selection>[-llama].conf`. The three name
components are the three axes, and each is independent of the other two:

| Component | Values | Parameter(s) | Meaning |
|---|---|---|---|
| `<cell>` | `uniform` `typ` `logp` `ent` `ent-only` `dis` `typ-dis` | `BETA`, `GAMMA`, `LAMBDA_D` | the reliability weighting — the axis this grid is about |
| `<budget>` | `unif` `adap` | `BUDGET_TAU` = `0.0` / `1.0` | uniform vs. perturbation-proportional head-wise budgets |
| `<selection>` | `topk` `div` | `LAMBDA_K`,`LAMBDA_V` = `0.0` / `0.1` | plain top-K vs. diversity-aware greedy selection |

7 × 2 × 2 × 2 models = **56 configs**. `unif-topk` is the paper's core configuration;
all main-text results use it.

#### The reliability cells

| `<cell>` | `(BETA, GAMMA, LAMBDA_D)` | Stage | Role |
|---|:---:|:---:|---|
| `uniform` | `(0, 0, 0)` | 1 | uniform `omega_m = 1/M` — **the missing control** |
| `typ` | `(1, 0, 0)` | 1 | pure typicality, entropy channel **off** — the real no-entropy control |
| `logp` | `(1, 1, 0)` | 1 | relabels the published "log-probability only" row exactly |
| `ent` | `(1, 1.3, 0)` | 1 | relabels the published LORE-KV /E core exactly |
| `ent-only` | `(0, 1, 0)` | 1 | pure entropy — run only if D1 shows `Var(Hbar)` dominating |
| `dis` | `(1, 1, 0.3)` | 2 | relabels the published LORE-KV /D core exactly |
| `typ-dis` | `(1, 0, 0.3)` | 2 | disagreement **instead of** entropy — what /D was meant to be |

`uniform`, `typ` and `typ-dis` were **unreachable** in the old `(LAMBDA_H, eta)`
parametrization, where `LAMBDA_H = 0` already meant `GAMMA = 1`.

#### Output-folder naming

A run writes to `experiments/<BENCHMARK>/pred/<name>`, where `<name>` mirrors the
config name so a folder is identifiable at a glance:

```
lorekv-<cell>-<brule>-<sel>[-<model>]_B_<B>_chunk_<cs>_M_<M>_T_<T>_lL_<lL>_lT_<lT>_nrm_<0|1>_alpha_<alpha>_kappa_<kappa>
└────────── identity: same string as its .conf ──────────┘└──── everything the config name does not pin ────┘
```

for example

```
lorekv-ent-unif-topk_B_128_chunk_4_M_4_T_8_lL_1.0_lT_1.0_nrm_0_alpha_0.0_kappa_1.0
lorekv-typ-dis-adap-div-llama_B_512_chunk_4_M_8_T_8_lL_1.0_lT_1.0_nrm_0_alpha_0.0_kappa_1.0
```

Baselines (`andpro`, `critical`, `causal_forward_only`) keep
`<mode>_<MODEL_TAG>_<B>_chunk_<cs>_red_<red>`.

Two properties worth knowing:

* **`<cell>`, `<brule>` and `<sel>` come from the numbers actually used, not from the
  config filename.** A CLI sweep such as
  `LAMBDA_D="0.0 0.3" CONFIG_FILE=config/lorekv-typ-dis-unif-topk.conf` writes one
  `lorekv-typ-...` folder and one `lorekv-typ-dis-...` folder, so a folder name can
  never disagree with what was run. Coefficients matching no named cell fall back to
  `b<beta>g<gamma>lD<lambdaD>` (e.g. `lorekv-b1g0lD0.1-unif-topk_...`).
* **One module generates and parses these names**: [`scripts/run_naming.py`](scripts/run_naming.py).
  `run.sh` calls it as a CLI; `collect_results.py`, `eval.py` and
  `report_length_time.py` import `parse_run_name` from it. It also reads the two
  pre-reparametrization schemes and translates them through Proposition 1, so tables
  can mix old and new folders.

**Renaming folders written before this change:**

```bash
python scripts/rename_pred_runs.py            # dry run — prints the plan
python scripts/rename_pred_runs.py --apply    # rename
```

It is idempotent, refuses to run if any two folders would collide, leaves baselines
alone, and renames symlinked (archived) runs in place without touching their targets.
Note that it relabels an old `lH_0.0` run as `logp` — `(1, 1, 0)`, a full unit of
entropy penalty — not as a no-penalty control, per Proposition 1.

#### The two sweeps

Hold `<budget>-<selection>` fixed and vary `<cell>` — that is the single-factor
**reliability** comparison, and it is the one the reparametrization makes well-posed:

```bash
for cell in uniform typ logp ent ent-only; do
  EXPERIMENT=LongBench BUDGET=128 \
    CONFIG_FILE="config/lorekv-$cell-unif-topk.conf" bash scripts/run_h100.sh
done
```

Hold `<cell>` fixed and vary `<budget>-<selection>` — that prices **allocation and
selection** at a fixed reliability rule (the old Group A/B/C axes):

```bash
for v in unif-topk adap-topk unif-div adap-div; do
  EXPERIMENT=LongBench BUDGET=128 \
    CONFIG_FILE="config/lorekv-ent-$v.conf" bash scripts/run_h100.sh
done
```

#### The lookahead-budget axis has no configs of its own

`NUM_PARTICLES` is sweepable from the environment and CLI beats the config file, so
the `M` ablation is an override rather than a separate family:

```bash
NUM_PARTICLES=1       CONFIG_FILE=config/lorekv-ent-unif-topk.conf bash scripts/run_h100.sh
NUM_PARTICLES="2 4 8" CONFIG_FILE=config/lorekv-ent-unif-topk.conf bash scripts/run_h100.sh
```

Reliability weighting is **vacuous at `M = 1`** (a single `R_m` is normalized away by
the softmax) and `LAMBDA_D` is **identically 0 at `M <= 2`**, so those rows are
insensitive to `<cell>`.

#### Retired: the `tlckv-*` family

The `tlckv-{pe,p,pd,m1}-*` configs have been removed from `config/` — every one of them
is a point of the matrix above:

| Retired | Equivalent |
|---|---|
| `tlckv-pe-m4-<b>-<s>` | `lorekv-ent-<b>-<s>` — `(1, 1.3, 0)` |
| `tlckv-p-m4-<b>-<s>` | `lorekv-logp-<b>-<s>` — `(1, 1, 0)`, **not** a no-penalty control |
| `tlckv-pd-m4-<b>-<s>` | `lorekv-dis-<b>-<s>` — `(1, 1, 0.3)`, disagreement *on top of* entropy |
| `tlckv-m1-<b>-<s>` | `NUM_PARTICLES=1` on any cell |
| `tlckv-*-mx-<b>-<s>` | `NUM_PARTICLES="2 4 8"` on any cell |

Note that `tlckv-pd-*` vs `tlckv-pe-*` was never a single-factor comparison of the
reliability term: `(1, 1, 0.3)` and `(1, 1.3, 0)` differ in **two** coefficients. The
single-factor test is `lorekv-typ-<b>-<s>` vs `lorekv-typ-dis-<b>-<s>`.

The retired files were moved (not deleted) to
`$SCRATCH/train_free_kv_archive/config-legacy/`. The `(LAMBDA_H, eta) -> (BETA, GAMMA,
LAMBDA_D)` shim in `run_h100.sh` is unchanged, so if you restore one it still runs and
still reproduces its published numbers.

---

## Running the reparametrized experiments

Everything below assumes the repo root and a live venv (`scripts/run_h100.sh` sources
`.env` and activates `$VENV_PATH` itself).

### 0. One-time setup on ACES

```bash
module purge && module load WebProxy GCC/11.3.0 CUDA/12.1.1
export REPO_ROOT="$PWD" SCRATCH="${SCRATCH:-/scratch/user/$USER}"
```

### 1. Stage 1 — the `(BETA, GAMMA)` plane at `LAMBDA_D = 0`

The five Stage-1 cells, LongBench, Mistral, `B=128` (the budget with resolving power):

```bash
for cell in uniform typ logp ent ent-only; do
  cfg="lorekv-$cell-unif-topk"; echo "===== $cfg ====="
  EXPERIMENT=LongBench CONFIG_FILE="$REPO_ROOT/config/$cfg.conf" BUDGET=128 \
    bash "$REPO_ROOT/scripts/run_h100.sh"
done
```

Same on RULER, and on Llama:

```bash
for cell in uniform typ logp ent ent-only; do
  EXPERIMENT=RULER CONFIG_FILE="$REPO_ROOT/config/lorekv-$cell-unif-topk.conf" BUDGET=128 \
    bash "$REPO_ROOT/scripts/run_h100.sh"
done

for cell in uniform typ logp ent ent-only; do
  EXPERIMENT=LongBench CONFIG_FILE="$REPO_ROOT/config/lorekv-$cell-unif-topk-llama.conf" \
    BUDGET=128 bash "$REPO_ROOT/scripts/run_h100.sh"
done
```

As a **SLURM batch job** instead (one submission per benchmark):

```bash
EXPERIMENT=LongBench BUDGET=128 sbatch scripts/run_h100_aces.sh
EXPERIMENT=RULER     BUDGET=128 sbatch scripts/run_h100_aces.sh
```

`scripts/run_h100_aces.sh` has **two modes, selected by whether `CONFIG_FILE` is set**:

*No `CONFIG_FILE`* — sweep `CELLS x VARIANTS x MODELS`, each overridable at submit
time (defaults: the five Stage-1 cells, `unif-topk` only, both models = 10 runs):

```bash
# the default sweep
EXPERIMENT=LongBench BUDGET=128 sbatch scripts/run_h100_aces.sh

# the full budget x selection cross at every Stage-1 cell (40 runs)
sbatch --export=ALL,VARIANTS="unif-topk adap-topk unif-div adap-div" scripts/run_h100_aces.sh

# Stage 2, Mistral only
sbatch --export=ALL,CELLS="dis typ-dis",MODELS=mistral scripts/run_h100_aces.sh
```

*With `CONFIG_FILE`* — run exactly that one config and **skip the sweep**. A bare
config name works as well as a path:

```bash
sbatch --export=ALL,CONFIG_FILE=lorekv-typ-unif-topk scripts/run_h100_aces.sh
sbatch --export=ALL,CONFIG_FILE=config/lorekv-typ-unif-topk.conf scripts/run_h100_aces.sh

# a single config across the whole budget grid
sbatch --export=ALL,CONFIG_FILE=lorekv-ent-unif-topk,BUDGET="128 256 512 1024" \
       scripts/run_h100_aces.sh
```

A `CONFIG_FILE` that does not exist is a hard error rather than a fallback:
`run_h100.sh` only tests `-f "$CONFIG_FILE"` and otherwise warns and silently uses its
inline defaults, so a typo would quietly run the wrong configuration. An empty
`CONFIG_FILE=` counts as unset and sweeps.

The two published cells `lorekv-logp-unif-topk` and `lorekv-ent-unif-topk` should reproduce
`40.29 / 41.39 / 41.30 / 41.40` and `40.28 / 41.38 / 41.84 / 42.21` exactly; if they do
not, the `(BETA, GAMMA)` mapping is wrong before anything else is read.

**Three seeds.** The gaps being resolved are 0.01–0.04, below any plausible noise floor
estimate. Run each cell three times and report the spread:

```bash
for seed in 0 1 2; do
  for cell in uniform typ logp ent; do
    EXPERIMENT=LongBench CONFIG_FILE="$REPO_ROOT/config/lorekv-$cell-unif-topk.conf" BUDGET=128 \
      PYTHONHASHSEED=$seed SEED=$seed MODEL_TAG="Mistral-7B-Instruct-v0.2-s$seed" \
      bash "$REPO_ROOT/scripts/run_h100.sh"
  done
done
```

### 2. Stage 2 — the `LAMBDA_D` probe (gated on test S4)

Run **only after** S4 shows a non-zero, sign-consistent semipartial correlation *per
benchmark*. At the best `(BETA*, GAMMA*)` from Stage 1:

```bash
for cell in dis typ-dis; do
  EXPERIMENT=LongBench CONFIG_FILE="$REPO_ROOT/config/lorekv-$cell-unif-topk.conf" BUDGET=128 \
    bash "$REPO_ROOT/scripts/run_h100.sh"
done
```

`LAMBDA_D` sweeps work the same way (`out_name` carries `_lD_<value>`, so runs never
collide):

```bash
EXPERIMENT=LongBench CONFIG_FILE="$REPO_ROOT/config/lorekv-typ-dis-unif-topk.conf" \
  LAMBDA_D="0.0 0.1 0.3 1.0" BUDGET=128 bash "$REPO_ROOT/scripts/run_h100.sh"
```

Calibrate `LAMBDA_D` so that `sd_m(LAMBDA_D * Dhat_m)` matches `sd_m(GAMMA * Hbar_m)`
before comparing the two channels — `Hbar` spans ~0–4 nats while `Dhat` is bounded by
`sqrt(M-1) ≈ 1.73` at `M=4`, so equal coefficients are **not** equal tilts.

### 3. Validation — run this before reading any reliability number

```bash
cd experiments/LongBench/figure_making_script/reparam_validation

# capture (one GPU pass; --utility adds the true-future-query decode for S3/S4)
python compute_validation.py \
  --config ../../../../config/lorekv-ent-unif-topk.conf \
  --benchmark longbench --dataset single --samples-per-dataset 8 \
  --num-particles 8 --budget 128 --utility

# report (no GPU)
python report_validation.py --data validation_data_<tag>.npz
```

RULER counterpart — run it separately, **never pooled**, since figure 10 already shows
`-Dhat_m` at `rho = -0.26` on LongBench and `+0.03` on RULER:

```bash
python compute_validation.py --config ../../../../config/lorekv-ent-unif-topk.conf \
  --benchmark ruler --dataset 4096 --samples-per-dataset 8 \
  --num-particles 8 --budget 128 --utility
```

| Test | What it checks | Failure means |
|---|---|---|
| **V1** | bootstrap CI of `mean(cbar_m)` contains 0 | positive ⇒ sampling law ≠ centering law (truncation/temperature). **Fix and re-capture before anything else.** |
| **V2** | logged `R_m` matches `BETA*cbar - GAMMA*Hbar - LAMBDA_D*Dhat` | an exact `T`-fold gap ⇒ the code used the sequence *total* where Eq. 3 uses the *mean* |
| **V3** | lag-`k` autocorrelation of `{c_u}` ≈ 0 | truncation leaked in, or a repetition regime; V4 no longer applies |
| **V4** | OLS slope of empirical `Var_m(cbar)` on `(1/T^2) sum_u V_u` ≈ 1 | Prop. 3 miscalibrated ⇒ report unnormalized `cbar` only |
| **V5** | across-prompt ratio `q95/q05` of `sqrt(Vbar/T)` | `< 1.3` keep `NORMALIZE_TYPICALITY=0`; `> 2` promote it |
| **D1** | share of `Var_m(cbar - Hbar)` carried by `Var(Hbar)` | if it dominates, the old "log-probability weighting" *was* an entropy penalty |
| **D2** | within-prompt Spearman(`cbar_m`, `Hbar_m`) | reports the confound directly |
| **S3** | Spearman of each signal vs per-particle oracle utility | the reading: does `cbar_m` outrank raw `log p`? |
| **S4** | semipartial Spearman(`D_m`, utility \| `cbar`, `Hbar`), ΔR², permutation null | **the gate for `LAMBDA_D > 0`.** Near-zero is a reportable negative control, not a failure. |

V1, V2, D1, D2 and S4 need only the trajectory means, which every existing capture
already stores (`cbar_m = logp_mean + H_mean`). V3, V4 and V5 need the per-step `c_u`
and the varentropy `V_u`, which is what the re-capture pass above adds — same `M*T`
decode, roughly one GPU-hour over 550 + 6,500 prompts.

### 4. Figures

The score-level figures now sweep the full `(BETA, GAMMA, LAMBDA_D)` grid on captured
particles (no re-decoding), so figures 1 and 10 answer S1/S3/S5 directly:

```bash
FIGURES="1 10" CONFIG_FILE=config/lorekv-ent-unif-topk.conf BUDGET=128 \
  BENCHMARK=longbench bash scripts/run_figures_aces.sh
```

### 5. Archiving finished runs to scratch

`pred/` folders are immutable once a run has finished and home is inode-limited, so
completed runs belong on `/scratch`. The helper moves them and leaves a symlink, so
`eval.py`, `collect_results.py` and the figure scripts need no change:

```bash
# dry run first — prints the plan, moves nothing
bash scripts/archive_runs_to_scratch.sh --pattern 'lorekv-ent-*'

# then apply
bash scripts/archive_runs_to_scratch.sh --pattern 'lorekv-ent-*' --apply
```

Because the folder name now leads with the run's identity, patterns read directly:

| Pattern | Selects |
|---|---|
| `'lorekv-ent-*'` | every entropy-weighted run, any budget rule / selection |
| `'lorekv-*-unif-topk_*'` | every core-configuration run, any reliability cell |
| `'lorekv-dis-*-llama_*'` | the Llama disagreement family |
| `'lorekv-*_B_1024_*'` | every `B=1024` run |

The entropy-weighted family (64 folders, ~77 MB across LongBench/RULER/NIAH) is
already archived to `$SCRATCH/train_free_kv_archive/<benchmark>/pred/`, symlinked back
under its current name.

Re-running is idempotent (an already-archived folder is a symlink, and the script
matches directories only). To undo one:
`rm <link> && mv $SCRATCH/train_free_kv_archive/<exp>/pred/<name> <link>`.

---

## Acknowledgement

We extend our gratitude to [AdaKV](https://github.com/FFY0/AdaKV) for their contributions of open-source code. This repository is built upon their latest code framework, with only minor modifications to the token score calculation.

## Citation
If you find this project helpful to your research, please consider citing our paper using the following BibTeX:
```
@inproceedings{
    geng2025accurate,
    title={Accurate {KV} Cache Eviction via Anchor Direction Projection for Efficient {LLM} Inference},
    author={Geng, Zijie and Wang, Jie and Liu, Ziqi and Ju, Feng and Li, Yiming and Li, Xing and Yuan, Mingxuan and Hao, Jianye and Lian, Defu and Chen, Enhong and Wu, Feng},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2025},
}
```