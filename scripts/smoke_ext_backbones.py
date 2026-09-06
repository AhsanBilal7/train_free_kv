#!/usr/bin/env python
"""End-to-end smoke test of the eviction pipeline on the models_ext backbones.

scripts/validate_models_ext.py checks that the vendored modelling code IS the
upstream model. This checks the other half: that each package's hijack actually
drives it -- prefill capture, lookahead rollout, compression and the flattened
varlen decode -- on Qwen3 and Ministral3, not just on Llama.

It runs the same setup pred.py does (same replace_* calls, same config_compress
fields, same flash_attention_2 load), on one long synthetic prompt, and prints
what comes out. A mode passes if it decodes non-empty text without raising; the
text itself is a sanity read, not an assertion.

    python scripts/smoke_ext_backbones.py --model "$SCRATCH/models/Qwen3-8B" \
        --modes causal_lookahead andpro critical --budget 128
"""

import argparse
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from models_ext import register_extra_architectures

register_extra_architectures()

from adaptive_snapkv.monkeypatch.monkeypatch import (
    replace_llama_adaptive, replace_ministral3_adaptive, replace_mistral_adaptive, replace_qwen3_adaptive,
)
from andpro_snapkv.monkeypatch.monkeypatch import (
    replace_llama_andpro, replace_ministral3_andpro, replace_mistral_andpro, replace_qwen3_andpro,
)
from causal_lookahead_snapkv.monkeypatch.lookahead_runner import generate_with_lookahead
from causal_lookahead_snapkv.monkeypatch.monkeypatch import (
    replace_llama_lookahead, replace_ministral3_lookahead, replace_mistral_lookahead, replace_qwen3_lookahead,
)

# Same grouping pred.py uses: `critical` and `ada` share the adaptive hijack and
# differ only in the criterion selected from model.config.mode at runtime.
PATCHERS = {
    "causal_lookahead": [replace_mistral_lookahead, replace_llama_lookahead,
                         replace_qwen3_lookahead, replace_ministral3_lookahead],
    "andpro": [replace_mistral_andpro, replace_llama_andpro,
               replace_qwen3_andpro, replace_ministral3_andpro],
    "critical": [replace_mistral_adaptive, replace_llama_adaptive,
                 replace_qwen3_adaptive, replace_ministral3_adaptive],
    "ada": [replace_mistral_adaptive, replace_llama_adaptive,
            replace_qwen3_adaptive, replace_ministral3_adaptive],
}

# A prompt long enough to leave real eviction candidates outside the protected set
# (first token + the 32-token observation window) at B=128, with one fact planted
# early that a correct decode has to recover from the retained cache.
NEEDLE = "The internal reference code for the Kestrel project is QX-7741."
FILLER = (
    "The archive contains routine maintenance notes. Equipment was inspected and "
    "returned to service. No anomalies were recorded during the reporting period. "
    "Scheduled calibration was completed on time and logged by the duty engineer. "
)


def build_prompt(tokenizer, target_tokens):
    body = FILLER * 4 + NEEDLE + " " + FILLER * 200
    ids = tokenizer(body, return_tensors="pt").input_ids[0][:target_tokens]
    body = tokenizer.decode(ids, skip_special_tokens=True)
    return (
        body
        + "\n\nQuestion: what is the internal reference code for the Kestrel project?\nAnswer:"
    )


def config_compress(model, budget, mode):
    """The fields pred.py's config_compress sets, at this repo's core configuration."""
    c = model.model.config
    c.window_size, c.base_capacity, c.kernel_size = 32, budget, 7
    c.pooling, c.floor_alpha, c.skip, c.normalize = "maxpool", 0.5, 0, None
    c.pyram_mode, c.pyram_beta = False, 20
    c.gqa_support, c.gqa_func = False, "mean"
    c.chunk_size, c.mode, c.lambda_red = 4, mode, 0.5
    c.lookahead_phase, c.lookahead_particle_idx = "collect", 0
    c.num_particles, c.lookahead_len, c.lookahead_temperature = 4, 8, 1.0
    c.particle_beta, c.particle_gamma, c.lambda_d = 1.0, 1.3, 0.0
    c.normalize_typicality = False
    c.robust_alpha, c.robust_topk_beta = 0.0, 1.0
    c.lambda_lookahead, c.lambda_k, c.lambda_v, c.budget_tau = 1.0, 0.0, 0.0, 0.0
    c.score_objective = "projected"


def run_mode(model_path, mode, budget, n_tokens, max_gen):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    for patch in PATCHERS[mode]:
        patch()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="flash_attention_2", trust_remote_code=True,
    ).eval()
    config_compress(model, budget, mode)

    prompt = build_prompt(tokenizer, n_tokens)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    n_ctx = inputs.input_ids.shape[-1]

    if mode == "causal_lookahead":
        ids = generate_with_lookahead(
            model, tokenizer, inputs.input_ids, inputs.attention_mask, max_gen,
            [tokenizer.eos_token_id],
        )
        text = tokenizer.decode(ids, skip_special_tokens=True)
    else:
        out = model.generate(**inputs, max_new_tokens=max_gen, num_beams=1, do_sample=False)
        text = tokenizer.decode(out[0][n_ctx:], skip_special_tokens=True)

    del model
    torch.cuda.empty_cache()
    return n_ctx, text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--modes", nargs="+", default=["causal_lookahead", "andpro", "critical"])
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--n-tokens", type=int, default=4000)
    ap.add_argument("--max-gen", type=int, default=32)
    args = ap.parse_args()

    failures = []
    for mode in args.modes:
        print(f"\n===== {os.path.basename(args.model)} · {mode} · B={args.budget} =====", flush=True)
        try:
            n_ctx, text = run_mode(args.model, mode, args.budget, args.n_tokens, args.max_gen)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            failures.append((mode, f"{type(exc).__name__}: {exc}"))
            continue
        print(f"  context {n_ctx} tokens -> {len(text)} chars")
        print(f"  answer: {text!r}")
        if not text:
            failures.append((mode, "empty generation"))
        elif "QX-7741" in text:
            print("  needle recovered")
        else:
            # Not a failure: at B=128 an eviction method is allowed to drop the fact.
            print("  needle NOT recovered (allowed at this budget; check it is not gibberish)")

    print("\n===== summary =====")
    for mode in args.modes:
        bad = [m for m, _ in failures if m == mode]
        print(f"  {mode:18s} {'FAIL' if bad else 'ok'}")
    for mode, why in failures:
        print(f"    {mode}: {why}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
