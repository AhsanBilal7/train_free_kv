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

### For LongBench
```bash
# Group A — Adaptive budget, Top-K selection
bash scripts/submit.sh tlckv-pe-m4-adap-topk
bash scripts/submit.sh tlckv-p-m4-adap-topk
bash scripts/submit.sh tlckv-m1-adap-topk

# Group B — Uniform budget, Top-K selection
bash scripts/submit.sh tlckv-pe-m4-unif-topk
bash scripts/submit.sh tlckv-p-m4-unif-topk
bash scripts/submit.sh tlckv-m1-unif-topk

# Group C — Adaptive budget, Diversity-aware selection
bash scripts/submit.sh tlckv-pe-m4-adap-div
bash scripts/submit.sh tlckv-p-m4-adap-div
bash scripts/submit.sh tlckv-m1-adap-div
```

### For RULER
```bash
# Group A — Adaptive budget, Top-K selection
EXPERIMENT=RULER bash scripts/submit.sh tlckv-pe-m4-adap-topk
EXPERIMENT=RULER bash scripts/submit.sh tlckv-p-m4-adap-topk
EXPERIMENT=RULER bash scripts/submit.sh tlckv-m1-adap-topk

# Group B — Uniform budget, Top-K selection
EXPERIMENT=RULER bash scripts/submit.sh tlckv-pe-m4-unif-topk
EXPERIMENT=RULER bash scripts/submit.sh tlckv-p-m4-unif-topk
EXPERIMENT=RULER bash scripts/submit.sh tlckv-m1-unif-topk

# Group C — Adaptive budget, Diversity-aware selection
EXPERIMENT=RULER bash scripts/submit.sh tlckv-pe-m4-adap-div
EXPERIMENT=RULER bash scripts/submit.sh tlckv-p-m4-adap-div
EXPERIMENT=RULER bash scripts/submit.sh tlckv-m1-adap-div
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

### Naming key

| Token | Parameter(s) | Value | Meaning |
|---|---|:---:|---|
| **PE** | `LAMBDA_H` | 0.3 | **Prob + Entropy** confidence: particles weighted by average log-probability penalised by predictive entropy (Eq. 7-8). Guards against fluent-but-uncertain continuations. |
| **P** | `LAMBDA_H` | 0.0 | **Prob-only** confidence: particles weighted by average log-probability alone; entropy ignored. |
| **M1** | `NUM_PARTICLES` | 1 | **Single particle**: one lookahead continuation. With M=1 the softmax weight is always 1.0, so the confidence signal has no effect — this isolates the future-query direction itself from the benefit of multiple diverse particles. |
| **Adap** | `BUDGET_TAU` | 1.0 | **Adaptive** head-wise budget: each head's KV budget is proportional to its total future-query perturbation mass (Eq. 19-20). Heads where deletions matter most get more tokens. |
| **Unif** | `BUDGET_TAU` | 0.0 | **Uniform** budget: every head receives an equal share of the layer budget, disabling adaptive allocation. |
| **TopK** | `LAMBDA_K`, `LAMBDA_V` | 0.0 | **Plain top-K** chunk selection: chunks are ranked by score alone, no diversity penalty. |
| **Div** | `LAMBDA_K`, `LAMBDA_V` | 0.1 | **Diversity-aware** greedy selection (Eq. 21): penalises chunks whose keys/values are already well represented by previously selected chunks. Useful for tasks with near-duplicate passages; keep values small so it only breaks ties. |

All nine configs also share: `NUM_PARTICLES=4` (except M1), `LOOKAHEAD_LEN=8`, `LAMBDA_LOOKAHEAD=1.0`, `LOOKAHEAD_TEMPERATURE=1.0`, `PARTICLE_ETA=1.0`, `ROBUST_ALPHA=0.0`, `ROBUST_TOPK_BETA=1.0`.

---

### Group A — Adaptive head-wise budget, plain top-K selection

Vary only the particle confidence signal. The adaptive budget focuses capacity on high-impact heads.

#### A1. `tlckv-pe-m4-adap-topk` — Prob+Entropy · 4 particles · Adaptive budget · Top-K

Entropy penalty down-weights uncertain particles (Eq. 7). Adaptive budget concentrates tokens on heads with the highest future-query perturbation mass (Eq. 20). Scores ranked by plain mean over window + particle queries.

```bash
CONFIG_FILE=config/tlckv-pe-m4-adap-topk.conf bash scripts/run_h100.sh
# or inline:
MODE=causal_lookahead NUM_PARTICLES=4 LOOKAHEAD_LEN=8 LAMBDA_LOOKAHEAD=1.0 \
LOOKAHEAD_TEMPERATURE=1.0 LAMBDA_H=0.3 PARTICLE_ETA=1.0 \
ROBUST_ALPHA=0.0 ROBUST_TOPK_BETA=1.0 LAMBDA_K=0.0 LAMBDA_V=0.0 \
BUDGET_TAU=1.0 BUDGET="128 256 512 1024" bash scripts/run_h100.sh
```

#### A2. `tlckv-p-m4-adap-topk` — Prob-only · 4 particles · Adaptive budget · Top-K

Removes the entropy penalty (Eq. 7) so particles are ranked purely by average log-probability.
Ablates the entropy component of confidence weighting vs. A1.

```bash
CONFIG_FILE=config/tlckv-p-m4-adap-topk.conf bash scripts/run_h100.sh
# or inline:
MODE=causal_lookahead NUM_PARTICLES=4 LOOKAHEAD_LEN=8 LAMBDA_LOOKAHEAD=1.0 \
LOOKAHEAD_TEMPERATURE=1.0 LAMBDA_H=0.0 PARTICLE_ETA=1.0 \
ROBUST_ALPHA=0.0 ROBUST_TOPK_BETA=1.0 LAMBDA_K=0.0 LAMBDA_V=0.0 \
BUDGET_TAU=1.0 BUDGET="128 256 512 1024" bash scripts/run_h100.sh
```

#### A3. `tlckv-m1-adap-topk` — Single particle · Adaptive budget · Top-K

One lookahead continuation; the confidence softmax collapses to weight 1.0 so entropy/log-prob
weighting is moot. Tests whether the future-query direction signal itself (from a single sample)
is sufficient, without the benefit of particle diversity.

```bash
CONFIG_FILE=config/tlckv-m1-adap-topk.conf bash scripts/run_h100.sh
# or inline:
MODE=causal_lookahead NUM_PARTICLES=1 LOOKAHEAD_LEN=8 LAMBDA_LOOKAHEAD=1.0 \
LOOKAHEAD_TEMPERATURE=1.0 LAMBDA_H=0.0 PARTICLE_ETA=1.0 \
ROBUST_ALPHA=0.0 ROBUST_TOPK_BETA=1.0 LAMBDA_K=0.0 LAMBDA_V=0.0 \
BUDGET_TAU=1.0 BUDGET="128 256 512 1024" bash scripts/run_h100.sh
```

---

### Group B — Uniform budget, plain top-K selection

Same confidence signals as Group A but with `BUDGET_TAU=0.0` (uniform per-head budget).
Ablates the contribution of adaptive head-wise budget allocation against Group A.

#### B1. `tlckv-pe-m4-unif-topk` — Prob+Entropy · 4 particles · Uniform budget · Top-K

```bash
CONFIG_FILE=config/tlckv-pe-m4-unif-topk.conf bash scripts/run_h100.sh
# or inline:
MODE=causal_lookahead NUM_PARTICLES=4 LOOKAHEAD_LEN=8 LAMBDA_LOOKAHEAD=1.0 \
LOOKAHEAD_TEMPERATURE=1.0 LAMBDA_H=0.3 PARTICLE_ETA=1.0 \
ROBUST_ALPHA=0.0 ROBUST_TOPK_BETA=1.0 LAMBDA_K=0.0 LAMBDA_V=0.0 \
BUDGET_TAU=0.0 BUDGET="128 256 512 1024" bash scripts/run_h100.sh
```

#### B2. `tlckv-p-m4-unif-topk` — Prob-only · 4 particles · Uniform budget · Top-K

```bash
CONFIG_FILE=config/tlckv-p-m4-unif-topk.conf bash scripts/run_h100.sh
# or inline:
MODE=causal_lookahead NUM_PARTICLES=4 LOOKAHEAD_LEN=8 LAMBDA_LOOKAHEAD=1.0 \
LOOKAHEAD_TEMPERATURE=1.0 LAMBDA_H=0.0 PARTICLE_ETA=1.0 \
ROBUST_ALPHA=0.0 ROBUST_TOPK_BETA=1.0 LAMBDA_K=0.0 LAMBDA_V=0.0 \
BUDGET_TAU=0.0 BUDGET="128 256 512 1024" bash scripts/run_h100.sh
```

#### B3. `tlckv-m1-unif-topk` — Single particle · Uniform budget · Top-K

```bash
CONFIG_FILE=config/tlckv-m1-unif-topk.conf bash scripts/run_h100.sh
# or inline:
MODE=causal_lookahead NUM_PARTICLES=1 LOOKAHEAD_LEN=8 LAMBDA_LOOKAHEAD=1.0 \
LOOKAHEAD_TEMPERATURE=1.0 LAMBDA_H=0.0 PARTICLE_ETA=1.0 \
ROBUST_ALPHA=0.0 ROBUST_TOPK_BETA=1.0 LAMBDA_K=0.0 LAMBDA_V=0.0 \
BUDGET_TAU=0.0 BUDGET="128 256 512 1024" bash scripts/run_h100.sh
```

---

### Group C — Adaptive budget, diversity-aware greedy selection

Same as Group A but with `LAMBDA_K=0.1 LAMBDA_V=0.1` (Eq. 21). The greedy selection rule
penalises chunks that are cosine-similar to already-selected chunks, reducing redundancy.
Ablates the diversity penalty against Group A.

> **Caution:** the cosine penalty can wrongly discard genuinely-important repeated evidence (e.g.
> the same entity across multiple supporting passages in multi-document QA). Values of 0.1 are
> deliberately small so redundancy only breaks ties between similarly-scored chunks.

#### C1. `tlckv-pe-m4-adap-div` — Prob+Entropy · 4 particles · Adaptive budget · Diversity

```bash
CONFIG_FILE=config/tlckv-pe-m4-adap-div.conf bash scripts/run_h100.sh
# or inline:
MODE=causal_lookahead NUM_PARTICLES=4 LOOKAHEAD_LEN=8 LAMBDA_LOOKAHEAD=1.0 \
LOOKAHEAD_TEMPERATURE=1.0 LAMBDA_H=0.3 PARTICLE_ETA=1.0 \
ROBUST_ALPHA=0.0 ROBUST_TOPK_BETA=1.0 LAMBDA_K=0.1 LAMBDA_V=0.1 \
BUDGET_TAU=1.0 BUDGET="128 256 512 1024" bash scripts/run_h100.sh
```

#### C2. `tlckv-p-m4-adap-div` — Prob-only · 4 particles · Adaptive budget · Diversity

```bash
CONFIG_FILE=config/tlckv-p-m4-adap-div.conf bash scripts/run_h100.sh
# or inline:
MODE=causal_lookahead NUM_PARTICLES=4 LOOKAHEAD_LEN=8 LAMBDA_LOOKAHEAD=1.0 \
LOOKAHEAD_TEMPERATURE=1.0 LAMBDA_H=0.0 PARTICLE_ETA=1.0 \
ROBUST_ALPHA=0.0 ROBUST_TOPK_BETA=1.0 LAMBDA_K=0.1 LAMBDA_V=0.1 \
BUDGET_TAU=1.0 BUDGET="128 256 512 1024" bash scripts/run_h100.sh
```

#### C3. `tlckv-m1-adap-div` — Single particle · Adaptive budget · Diversity

```bash
CONFIG_FILE=config/tlckv-m1-adap-div.conf bash scripts/run_h100.sh
# or inline:
MODE=causal_lookahead NUM_PARTICLES=1 LOOKAHEAD_LEN=8 LAMBDA_LOOKAHEAD=1.0 \
LOOKAHEAD_TEMPERATURE=1.0 LAMBDA_H=0.0 PARTICLE_ETA=1.0 \
ROBUST_ALPHA=0.0 ROBUST_TOPK_BETA=1.0 LAMBDA_K=0.1 LAMBDA_V=0.1 \
BUDGET_TAU=1.0 BUDGET="128 256 512 1024" bash scripts/run_h100.sh
```

---

### Quick-reference summary table

| # | Config file | Confidence | `NUM_PARTICLES` | `LAMBDA_H` | Budget | `BUDGET_TAU` | Selection | `LAMBDA_K/V` |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A1 | `tlckv-pe-m4-adap-topk` | Prob+Entropy | 4 | 0.3 | Adaptive | 1.0 | Top-K | 0.0 |
| A2 | `tlckv-p-m4-adap-topk` | Prob-only | 4 | 0.0 | Adaptive | 1.0 | Top-K | 0.0 |
| A3 | `tlckv-m1-adap-topk` | — (M=1) | 1 | 0.0 | Adaptive | 1.0 | Top-K | 0.0 |
| B1 | `tlckv-pe-m4-unif-topk` | Prob+Entropy | 4 | 0.3 | Uniform | 0.0 | Top-K | 0.0 |
| B2 | `tlckv-p-m4-unif-topk` | Prob-only | 4 | 0.0 | Uniform | 0.0 | Top-K | 0.0 |
| B3 | `tlckv-m1-unif-topk` | — (M=1) | 1 | 0.0 | Uniform | 0.0 | Top-K | 0.0 |
| C1 | `tlckv-pe-m4-adap-div` | Prob+Entropy | 4 | 0.3 | Adaptive | 1.0 | Diversity | 0.1 |
| C2 | `tlckv-p-m4-adap-div` | Prob-only | 4 | 0.0 | Adaptive | 1.0 | Diversity | 0.1 |
| C3 | `tlckv-m1-adap-div` | — (M=1) | 1 | 0.0 | Adaptive | 1.0 | Diversity | 0.1 |

All configs share: `ROBUST_ALPHA=0.0`, `ROBUST_TOPK_BETA=1.0` (plain mean aggregation), `PARTICLE_ETA=1.0`, `LOOKAHEAD_LEN=8`, `LAMBDA_LOOKAHEAD=1.0`, `LOOKAHEAD_TEMPERATURE=1.0`.

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