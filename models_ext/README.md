# `models_ext` — backbones the pinned Transformers cannot load

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
and two of the backbones the paper reports were released after that pin:

| backbone | `model_type` | needs |
|---|---|---|
| `Qwen/Qwen3-8B` | `qwen3` | transformers ≥ 4.51 |
| `mistralai/Ministral-3-8B-Instruct-2512` | `mistral3` / `ministral3` | transformers 5.x |

Upgrading transformers to reach them would invalidate all four hijacks at once, on
backbones whose results are already in the paper. So instead the two architectures
are re-expressed here in the 4.44 Llama idiom. Both subclass
`transformers.models.llama.modeling_llama`, which means they expose exactly the
surface the hijacks patch and inherit Llama's cache, causal mask, rotary plumbing
and generation input prep unchanged.

## What is actually different from Llama

Almost nothing, which is why this is tractable.

**Qwen3** (`modeling_qwen3.py`)
* per-head RMSNorm on q and k, after the head split and before RoPE;
* `head_dim` read from the config rather than `hidden_size // num_heads`.

**Ministral3** (`modeling_ministral3.py`)
* a YaRN inverse-frequency schedule. 4.44 *has* a `"yarn"` branch, but an older
  one: it takes the correction range from `max_position_embeddings` (262144)
  instead of `original_max_position_embeddings` (16384), and derives the attention
  factor as `0.1*log(factor)+1` instead of from the mscale ratio. Using it would
  silently mis-rotate every position, so `compute_yarn_inv_freq` reimplements the
  transformers 5.16 formula;
* a post-RoPE query scale `1 + beta*log(1 + floor(pos/16384))` (the "llama-4"
  length scaling). It is exactly 1.0 below 16384 tokens and 1.069 in the
  16384–32767 band.

Both hooks (`project_qkv`, `scale_query`) live on the attention class, so each
eviction package needs **one** extra function — the attention forward — rather than
a whole new hijack file. Those are generated from the package's Llama hijack by
`scripts/gen_ext_hijacks.py`, so they cannot drift from it.

## Checkpoint preparation

Neither backbone is loaded straight from the Hub.

`scripts/prepare_qwen3_local.py` writes `$SCRATCH/models/Qwen3-8B`: weights
symlinked, everything else copied, and `tokenizer.json` rewritten from the
`["a","b"]` merge format (tokenizers ≥ 0.20) to the `"a b"` form that tokenizers
0.19 parses. Without that rewrite the tokenizer load dies with
`data did not match any variant of untagged enum ModelWrapper`. A local directory
also sidesteps a cache-layout trap: `.env` points `TRANSFORMERS_CACHE` at `$HF_HOME`
while `huggingface_hub` downloads into `$HF_HOME/hub`, and ACES compute nodes have
no outbound network to cover the difference.

`scripts/convert_ministral3.py` writes `$SCRATCH/models/Ministral-3-8B-Instruct-2512`:
the published checkpoint is a `Mistral3ForConditionalGeneration` (Pixtral vision
tower + `ministral3` text model, every text weight under a `language_model.`
prefix), and the script keeps the text tensors, renames them, drops the vision
tower and the projector, and flattens `config.text_config` into a standalone
config. Convert the **`-BF16`** repo: the default one is FP8-quantized.

## Correctness

`scripts/validate_models_ext.py` runs the same checkpoint on the same tokens under
both implementations, in two processes because they need different transformers
releases, and compares logits. In **float32 with eager attention** the two are
expected to agree exactly; bfloat16 with sdpa does not, because torch 2.1 and
torch 2.6 select different sdpa backends and the disagreement on the largest
logits reaches ~12 while top-1 still agrees on 99.5% of positions. That is kernel
noise, not a modelling difference, and float32/eager is what settles the question.

Recorded results (Qwen3-8B, 256 tokens, float32, eager):

```
  max |a-b|          0.00000
  mean cosine        1.00000000
  top-1 agreement    100.0000%  (256/256 positions)
```

The Qwen3 tokenizer rewrite is checked separately: the rewritten and original
files produce identical ids on a mixed ASCII/unicode/whitespace/code probe, and
`apply_chat_template(..., enable_thinking=False)` returns the identical string
under both stacks.

## Evaluation-protocol notes

* **Qwen3 runs in non-thinking mode.** It is a hybrid reasoning model; under the
  default template it opens a `<think>` block and spends the whole short LongBench
  budget inside it, so every answer comes back empty. `pred.py`'s `build_chat`
  passes `enable_thinking=False`, the template's own switch, which emits an
  already-closed empty think block.
* **Ministral3 uses plain `[INST]…[/INST]`**, not the packaged chat template. That
  template injects a multi-hundred-token "Le Chat" system prompt (with the current
  date and tool-use instructions) ahead of every prompt, which would change the
  long-context input the eviction methods are being compared on.
