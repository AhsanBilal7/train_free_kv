#!/usr/bin/env python
"""End-to-end test of the eviction pipeline on the models_ext backbones (Qwen2, Ministral3).

``scripts/validate_models_ext.py`` checks that the Ministral3 implementation is
equal to the upstream model. This script checks that each package drives the
backbone: prefill, lookahead, compression and varlen decoding.

The script uses the same :mod:`lorekv.harness` calls as pred.py on one long
synthetic prompt and prints the result. A mode passes if it decodes non-empty
text without an error.

Usage::

    python scripts/smoke_ext_backbones.py --model models/Qwen2.5-14B-Instruct \\
        --modes causal_lookahead andpro critical --budget 128
"""

import argparse
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lorekv.harness import add_method_args, configure_model, install_method  # noqa: E402
from lorekv.monkeypatch.lookahead_runner import generate_with_lookahead  # noqa: E402
from models_ext import register_extra_architectures  # noqa: E402

register_extra_architectures()

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
    # Repeat until the filler alone exceeds the target, so --n-tokens is honoured
    # for every tokenizer rather than silently capping at whatever one pass yields.
    """Return a prompt of about ``target_tokens`` tokens with one fact near the start."""
    reps = max(200, target_tokens // 40)
    body = FILLER * 4 + NEEDLE + " " + FILLER * reps
    ids = tokenizer(body, return_tensors="pt").input_ids[0][:target_tokens]
    body = tokenizer.decode(ids, skip_special_tokens=True)
    return (
        body
        + "\n\nQuestion: what is the internal reference code for the Kestrel project?\nAnswer:"
    )


def config_compress(model, budget, mode):
    """Apply the settings of pred.py with the core LORE-KV configuration (Table 4)."""
    args = add_method_args(argparse.ArgumentParser()).parse_args(
        ["--mode", mode, "--budget", str(budget)])
    configure_model(model, args)


def run_mode(model_path, mode, budget, n_tokens, max_gen):
    """Decode the synthetic prompt with one mode and return the context length, the text and the peak memory."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    install_method(mode)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="flash_attention_2", trust_remote_code=True,
    ).eval()
    config_compress(model, budget, mode)

    prompt = build_prompt(tokenizer, n_tokens)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    n_ctx = inputs.input_ids.shape[-1]
    torch.cuda.reset_peak_memory_stats()

    if mode == "causal_lookahead":
        ids = generate_with_lookahead(
            model, tokenizer, inputs.input_ids, inputs.attention_mask, max_gen,
            [tokenizer.eos_token_id],
        )
        text = tokenizer.decode(ids, skip_special_tokens=True)
    else:
        out = model.generate(**inputs, max_new_tokens=max_gen, num_beams=1, do_sample=False)
        text = tokenizer.decode(out[0][n_ctx:], skip_special_tokens=True)

    peak = torch.cuda.max_memory_allocated() / 2**30
    del model
    torch.cuda.empty_cache()
    return n_ctx, text.strip(), peak


def prefill_equivalence(model_path, n_tokens):
    """Check that the patched prefill gives the same logits as the original model.

    The ``collect`` phase does not evict. It is causal attention that also records
    the window queries and the full K and V. Its logits must thus be equal to those
    of the original model. The check covers the Qwen2 rotary path, which must size
    its table from the positions and not from the cache.

    Both models load in the same process, so the patch is installed after the
    reference logits are computed.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    ids = tokenizer(build_prompt(tokenizer, n_tokens), return_tensors="pt").input_ids.to("cuda:0")

    def last_logits():
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
            attn_implementation="flash_attention_2", trust_remote_code=True,
        ).eval()
        config_compress(model, 128, "causal_lookahead")
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True)
        # The hijacked model forward keeps only the last position; index from the
        # end so the same expression works for both.
        v = out.logits[0, -1].float().cpu()
        del model, out
        torch.cuda.empty_cache()
        return v

    stock = last_logits()
    install_method("causal_lookahead")
    patched = last_logits()

    diff = (stock - patched).abs()
    agree = int(stock.argmax()) == int(patched.argmax())
    print(f"  stock vs patched prefill: max |diff| {diff.max():.6f}, "
          f"mean {diff.mean():.6f}, argmax {'match' if agree else 'DIFFER'}")
    return diff.max().item() == 0.0 and agree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--modes", nargs="+", default=["causal_lookahead", "andpro", "critical"])
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--n-tokens", type=int, default=4000)
    ap.add_argument("--max-gen", type=int, default=32)
    ap.add_argument("--skip-prefill-check", action="store_true")
    args = ap.parse_args()

    failures = []
    if not args.skip_prefill_check:
        print(f"\n===== {os.path.basename(args.model)} · patched-vs-stock prefill =====", flush=True)
        try:
            if not prefill_equivalence(args.model, args.n_tokens):
                failures.append(("prefill-equivalence", "patched prefill differs from stock"))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            failures.append(("prefill-equivalence", f"{type(exc).__name__}: {exc}"))
    for mode in args.modes:
        print(f"\n===== {os.path.basename(args.model)} · {mode} · B={args.budget} =====", flush=True)
        try:
            n_ctx, text, peak = run_mode(args.model, mode, args.budget, args.n_tokens, args.max_gen)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            failures.append((mode, f"{type(exc).__name__}: {exc}"))
            continue
        print(f"  context {n_ctx} tokens -> {len(text)} chars, peak GPU {peak:.1f} GiB")
        print(f"  answer: {text!r}")
        if not text:
            failures.append((mode, "empty generation"))
        elif "QX-7741" in text:
            print("  needle recovered")
        else:
            # Not a failure: at B=128 an eviction method is allowed to drop the fact.
            print("  needle NOT recovered (allowed at this budget; check it is not gibberish)")

    print("\n===== summary =====")
    if not args.skip_prefill_check:
        bad = [m for m, _ in failures if m == "prefill-equivalence"]
        print(f"  {'prefill-equivalence':18s} {'FAIL' if bad else 'ok'}")
    for mode in args.modes:
        bad = [m for m, _ in failures if m == mode]
        print(f"  {mode:18s} {'FAIL' if bad else 'ok'}")
    for mode, why in failures:
        print(f"    {mode}: {why}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
