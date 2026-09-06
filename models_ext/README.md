# `models_ext` — backbones beyond the pinned Transformers release

## Why this exists

Every eviction method in this repo is a monkeypatch. `adaptive_snapkv`,
`andpro_snapkv`, `causal_andpro_snapkv` and `causal_lookahead_snapkv` each replace

```
<Model>ForCausalLM.prepare_inputs_for_generation
<Model>FlashAttention2.forward
<Model>Model.forward
```

`*FlashAttention2` was deleted in transformers 4.48, when attention moved behind
`ALL_ATTENTION_FUNCTIONS`. The repo is therefore pinned to **transformers 4.44.2**,
and the two backbones added on top of Mistral-7B-Instruct-v0.2 and
Llama-3.1-8B-Instruct each need something under that pin — but not the same thing.

| backbone | `model_type` | situation |
|---|---|---|
| `Qwen/Qwen2.5-14B-Instruct` | `qwen2` | **native to 4.44**, no vendoring; but its 4.44 implementation predates the rotary refactor, so it needs its own hijack forwards |
| `mistralai/Ministral-3-8B-Instruct-2512` | `mistral3` / `ministral3` | added in transformers 5.x; **vendored** here |

Upgrading transformers to reach Ministral-3 would invalidate all four hijacks at
once, on backbones whose results are already in the paper. So it is re-expressed
here in the 4.44 Llama idiom: `modeling_ministral3.py` subclasses
`transformers.models.llama.modeling_llama`, exposing exactly the surface the
hijacks patch and inheriting Llama's cache, causal mask, rotary plumbing and
generation input prep unchanged.

## What is actually different

**Ministral3** (`modeling_ministral3.py`), vs. Llama:
* a YaRN inverse-frequency schedule. 4.44 *has* a `"yarn"` branch, but an older
  one: it takes the correction range from `max_position_embeddings` (262144)
  instead of `original_max_position_embeddings` (16384), and derives the attention
  factor as `0.1*log(factor)+1` instead of from the mscale ratio. Using it would
  silently mis-rotate every position, so `compute_yarn_inv_freq` reimplements the
  transformers 5.16 formula;
* a post-RoPE query scale `1 + beta*log(1 + floor(pos/16384))` (the "llama-4"
  length scaling). It is exactly 1.0 below 16384 tokens and 1.069 in the
  16384–32767 band.

Both are exposed as hooks on the attention class (`project_qkv`, `scale_query`),
so each eviction package needs one extra attention forward and nothing else.

**Qwen2**, vs. Llama in 4.44: the attention forward takes no
`position_embeddings`; the rotary module is table-based (`rotary_emb(x, seq_len=…)`)
and `apply_rotary_pos_emb` indexes that table with `position_ids`; and the model
forward therefore neither computes nor threads `position_embeddings` through the
layers. So the package supplies a Qwen2 model forward as well as an attention one.

The one substantive decision in that port: **`seq_len` for the rotary table must
come from the positions being served, not from the cache.** In the post-eviction
phase the flattened per-head cache reports a length of 1 while `position_ids`
keeps counting, so a cache-derived length yields a table thousands of rows short.
The patched model forward computes `_rope_seq_len` once per call (reading it off a
device tensor is a synchronization, and 48 layers × every decode step adds up) and
the attention layers read it from the config.

All of these files are generated from each package's own Llama hijack by
`scripts/gen_ext_hijacks.py`, so they cannot drift from it.

## Checkpoint preparation

Neither backbone is loaded straight from the Hub.

`scripts/prepare_local_checkpoint.py` writes `$SCRATCH/models/<name>`: weights
symlinked, everything else copied, and `tokenizer.json` rewritten from the
`["a","b"]` merge format (tokenizers ≥ 0.20) to the `"a b"` form that tokenizers
0.19 parses, *when needed* — Qwen2.5-14B-Instruct already ships the old format and
is copied through untouched. A local directory also sidesteps a cache-layout trap:
`.env` points `TRANSFORMERS_CACHE` at `$HF_HOME` while `huggingface_hub` downloads
into `$HF_HOME/hub`, and ACES compute nodes have no outbound network to cover the
difference.

`scripts/convert_ministral3.py` writes
`$SCRATCH/models/Ministral-3-8B-Instruct-2512`: the published checkpoint is a
`Mistral3ForConditionalGeneration` (Pixtral vision tower + `ministral3` text model,
every text weight under a `language_model.` prefix), and the script keeps the text
tensors, renames them, drops the vision tower and the projector, and flattens
`config.text_config` into a standalone config. Convert the **`-BF16`** repo: the
default one is FP8-quantized.

## Correctness

Two checks, both scripted.

**1. Vendored vs. upstream** (`scripts/validate_models_ext.py`). Runs the same
checkpoint on the same tokens under both implementations — two processes, because
they need different transformers releases — and compares logits. In **float32 with
eager attention** the two are expected to agree exactly; bfloat16 with sdpa does
not, because torch 2.1 and torch 2.6 select different sdpa backends and the
disagreement on the largest logits reaches ~12 while top-1 still agrees on 99.5% of
positions. That is kernel noise, not a modelling difference, and float32/eager is
what settles the question.

Recorded (Ministral-3-8B, 256 tokens, float32, eager; upstream side loads the
original Mistral3 vision-language wrapper, so this validates the text-tower
extraction and the modelling code together):

```
  max |a-b|          0.00000
  mean cosine        1.00000000
  top-1 agreement    100.0000%  (256/256 positions)
```

**2. Patched vs. stock prefill** (`scripts/smoke_ext_backbones.py`). The eviction
pipeline's "collect" phase evicts nothing, so its logits must equal the unpatched
model's. This is what exercises the rewritten Qwen2 rotary path end to end.
Recorded for both backbones: `max |diff| 0.000000, argmax match`. The same script
then decodes a 4k-token needle prompt under `causal_lookahead`, `andpro` and
`critical` at B=128 and checks the answer comes back.

## Evaluation-protocol notes

* **Qwen2.5-Instruct uses its ChatML template** (`pred.py`'s `build_chat`), which
  prepends the model's own ~20-token default system message. Without the template
  the model continues the document instead of answering. `enable_thinking=False` is
  passed and is inert for Qwen2.5 — its template does not read the variable — but
  guards against a hybrid-reasoning Qwen checkpoint silently spending the whole
  short generation budget inside a `<think>` block.
* **Ministral3 uses plain `[INST]…[/INST]`**, not the packaged chat template. That
  template injects a multi-hundred-token "Le Chat" system prompt (with the current
  date and tool-use instructions) ahead of every prompt, which would change the
  long-context input the eviction methods are being compared on.
* **Qwen2.5-14B is capped at `MAX_LENGTH = 31500`**, where the other backbones run
  at the repo default of 64000 (i.e. no truncation). Two reasons, both forcing:
  its context window is 32768 with no RoPE scaling, and LongBench's narrativeqa
  reaches ~64k tokens, so the default would score half that task outside the range
  the model was trained for; and the memory does not fit. Measured peak GPU
  allocation for `causal_lookahead` (the dominant mode -- the baselines peak around
  32 GiB because they never clone the prefill cache):

  | context | peak |
  |---|---|
  | 7,585 | 36.2 GiB |
  | 18,685 | 47.6 GiB |
  | 26,085 | 55.2 GiB |

  That is 28.4 GiB of weights plus 1.03 GiB per 1000 tokens, so 31,500 tokens needs
  ~61 GiB and fits an 80 GB H100 with ~18 GiB spare, while 64,000 would need ~94 GiB
  and cannot run at all. `MAX_LENGTH` is a config knob (`experiments/LongBench/run.sh`)
  that defaults to the old 64000, so no previously reported run is affected. The cap
  applies to every method equally, so the comparison *within* this backbone is
  unaffected; across backbones it is a documented difference in input length.
* Qwen2.5-14B is roughly twice the parameters of the other three backbones, so it
  is also correspondingly slower per sample.
