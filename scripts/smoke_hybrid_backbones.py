#!/usr/bin/env python
"""Check the hybrid-attention patches without a reference environment.

The ``collect`` phase does not evict. A correct patch thus gives the same prefill
logits as the original model. A difference shows an error in the patched
attention forward, for example a missing gate, norm or rotary call.

Run in float32 with eager attention. In bfloat16 a correct patch can show logit
differences of about 12 (99.5% top-1 agreement), because different SDPA kernels
are selected. In float32 with eager attention a correct patch gives equal logits.

Usage::

    python scripts/smoke_hybrid_backbones.py --model models/Qwen3.5-9B
    python scripts/smoke_hybrid_backbones.py --model models/gemma-4-12B-it
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from lorekv.monkeypatch.hybrid_support import (
        hybrid_manifest, is_hybrid)

    cfg = AutoConfig.from_pretrained(args.model)
    mt = getattr(cfg, "model_type", "")
    print(f"model_type={mt}  hybrid={is_hybrid(cfg)}")
    man = hybrid_manifest(cfg, base_capacity=128)
    print(f"  compressible layers ({man['num_compressed_layers']}/{man['num_layers']}): "
          f"{man['compressed_layers']}")
    for t, d in man["skipped_layer_types"].items():
        print(f"  skipping {d['count']:3d} x {t}: {d['reason']}")
    print(f"  cache share held by compressed layers: {man['cache_share_of_compressed_layers_at_n']}")

    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = torch.arange(10, 10 + args.tokens).unsqueeze(0)

    def load():
        m = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, attn_implementation="eager", device_map="cuda")
        return m.eval()

    print("\n[1/2] unpatched reference prefill ...")
    model = load()
    with torch.no_grad():
        ref = model(ids.to(model.device)).logits.float().cpu()
    del model
    torch.cuda.empty_cache()

    print("[2/2] patched prefill (collect phase evicts nothing) ...")
    from lorekv.monkeypatch.monkeypatch import (
        replace_gemma4_lookahead, replace_qwen3_5_lookahead)
    if mt == "qwen3_5":
        replace_qwen3_5_lookahead()
    elif mt in ("gemma4_unified", "gemma4"):
        replace_gemma4_lookahead()
    else:
        raise SystemExit(f"no hybrid hijack registered for model_type={mt!r}")

    model = load()
    # Minimal config the kv_cluster asserts on; values are irrelevant because the
    # collect phase does not evict -- they only have to exist.
    for k, v in dict(window_size=32, base_capacity=128, chunk_size=4, skip=0,
                     num_particles=0, lookahead_len=0, robust_alpha=0.0,
                     robust_topk_beta=1.0, lambda_lookahead=1.0, lambda_k=0.0,
                     lambda_v=0.0, budget_tau=0.0, lookahead_temperature=1.0,
                     lookahead_phase="collect").items():
        setattr(model.config, k, v)
        tc = getattr(model.config, "text_config", None)
        if tc is not None:
            setattr(tc, k, v)
    with torch.no_grad():
        got = model(ids.to(model.device)).logits.float().cpu()

    diff = (ref - got).abs().max().item()
    agree = (ref.argmax(-1) == got.argmax(-1)).float().mean().item()
    print(f"\n  max |logit diff| = {diff:.3e}")
    print(f"  top-1 agreement  = {100*agree:.3f}%")
    tol = 0.0 if args.dtype == "float32" else 5e-2
    ok = diff <= tol if args.dtype == "float32" else agree > 0.99
    print("  RESULT:", "PASS -- patched prefill is identical" if ok else "FAIL -- hijack changes the model")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
