#!/usr/bin/env python
"""Run prefill, lookahead, compression and ragged decoding on a small random hybrid model.

The script uses the same LORE-KV and baseline code as a real hybrid-backbone run,
on a random Qwen3.5 model (2 full-attention and 6 linear-attention layers) or a
random Gemma 4 model with small dimensions. One run takes seconds on CPU. It
requires the transformers 5.x environment of the hybrid backbones.

The test checks the data flow, not the quality: random weights give random
tokens. It checks that the pipeline finishes and that eviction occurs.

Usage::

    python scripts/harness_hybrid_cycle.py              # Qwen3.5 shape
    python scripts/harness_hybrid_cycle.py --arch gemma4
"""

import argparse
import os
import sys
import traceback

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# Only the config (layer_types, rope settings) is read from these checkpoints; the
# weights are never loaded. Override with --base-config.
BASE_CONFIGS = {"qwen35": os.path.join(REPO_ROOT, "models", "Qwen3.5-9B"),
                "gemma4": os.path.join(REPO_ROOT, "models", "gemma-4-12B-it")}


def build_tiny(arch, base_config=None):
    """Make a small hybrid model with the same layer-type structure as the real model."""
    from transformers import AutoConfig, AutoModelForCausalLM

    if arch == "qwen35":
        base = AutoConfig.from_pretrained(base_config or BASE_CONFIGS["qwen35"])
        t = base.text_config
        # 8 layers keeping the 3:1 linear:full ratio, so layer_types is realistic.
        t.num_hidden_layers = 8
        t.layer_types = ["linear_attention"] * 3 + ["full_attention"] \
                        + ["linear_attention"] * 3 + ["full_attention"]
        t.hidden_size = 256
        t.intermediate_size = 512
        t.num_attention_heads = 4
        t.num_key_value_heads = 2
        t.head_dim = 32
        t.linear_key_head_dim = 16
        t.linear_value_head_dim = 16
        t.linear_num_key_heads = 4
        t.linear_num_value_heads = 8
        t.vocab_size = 512
        base.vocab_size = 512
        cfg = base
    else:
        base = AutoConfig.from_pretrained(base_config or BASE_CONFIGS["gemma4"])
        t = base.text_config
        # Take a PREFIX of the real layer_types rather than inventing a pattern:
        # Gemma 4's config carries per-layer rope settings keyed by layer type, and a
        # hand-written list disagrees with them ("Layer type 'sliding_attention' is not
        # homogeneous across layers"). The real list is 5 sliding : 1 full, so 12
        # layers gives two compressible ones at indices 5 and 11.
        n_keep = 12
        t.layer_types = list(t.layer_types)[:n_keep]
        t.num_hidden_layers = n_keep
        for attr in ("per_layer_config",):
            v = getattr(t, attr, None)
            if isinstance(v, (list, tuple)) and len(v) > n_keep:
                setattr(t, attr, type(v)(list(v)[:n_keep]))
        t.hidden_size = 256
        t.intermediate_size = 512
        t.num_attention_heads = 4
        # num_key_value_heads and head_dim live in per_layer_config for Gemma 4, so
        # setting them on the text config alone leaves the MODEL built from the
        # per-layer values (8 / 256) while the config reports the overrides -- the
        # tensors then mismatch ("size of tensor a (4) must match tensor b (2)").
        # Write both, or the tiny harness model is internally inconsistent.
        t.num_key_value_heads = 2
        t.head_dim = 32
        # Reproduce Gemma 4's ASYMMETRIC geometry, do not flatten it: the real
        # checkpoint has full-attention layers at 1 KV head x head_dim 512 and
        # sliding layers at 8 x 256. Setting one uniform value made the harness
        # test an architecture that does not exist -- it passed while production
        # produced garbage. Scale both down but keep the 1-vs-8 / 2x-width ratio.
        plc = getattr(t, "per_layer_config", None)
        if plc is not None:
            for i in range(t.num_hidden_layers):
                try:
                    e = plc[i]
                    if t.layer_types[i] == "full_attention":
                        e.num_key_value_heads = 1
                        e.head_dim = 64
                    else:
                        e.num_key_value_heads = 4
                        e.head_dim = 32
                except Exception:
                    break
        t.sliding_window = 16
        t.vocab_size = 512
        base.vocab_size = 512
        cfg = base

    cfg._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_config(cfg)
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="qwen35", choices=["qwen35", "gemma4"])
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--particles", type=int, default=2)
    ap.add_argument("--lookahead", type=int, default=3)
    ap.add_argument("--gen", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--base-config", default=None,
                    help="checkpoint dir or Hub id whose config shapes the tiny model")
    ap.add_argument("--method", default="lookahead",
                    choices=["lookahead", "andpro", "adaptive"],
                    help="adaptive == the CriticalKV baseline")
    args = ap.parse_args()

    from lorekv.monkeypatch.monkeypatch import (
        replace_gemma4_lookahead, replace_qwen3_5_lookahead)
    from lorekv.monkeypatch.hybrid_support import compressible_layers

    if args.method == "lookahead":
        (replace_qwen3_5_lookahead if args.arch == "qwen35" else replace_gemma4_lookahead)()
    else:
        from lorekv.monkeypatch.hybrid_baseline_hijack import (
            replace_hybrid_baseline)
        replace_hybrid_baseline(args.method)
    model = build_tiny(args.arch, args.base_config).to(args.device)

    comp = compressible_layers(model.config)
    tcfg = getattr(model.config, "text_config", None) or model.config
    print(f"arch={args.arch} layers={tcfg.num_hidden_layers} compressible={comp}")

    for k, v in dict(window_size=8, base_capacity=args.budget, chunk_size=4, skip=0,
                     num_particles=args.particles, lookahead_len=args.lookahead,
                     robust_alpha=0.0, robust_topk_beta=1.0, lambda_lookahead=1.0,
                     lambda_k=0.0, lambda_v=0.0, budget_tau=0.0,
                     lookahead_temperature=1.0, particle_beta=1.0, particle_gamma=1.3,
                     lambda_d=0.0, normalize_typicality=False,
                     score_objective="projected", lookahead_phase="collect",
                     # SnapKV-family keys the baselines assert on;
                     # lorekv.harness.configure_model supplies these in a real run.
                     kernel_size=7, pooling="maxpool", floor_alpha=0.5,
                     normalize=False, pyram_mode=False, pyram_beta=20,
                     gqa_support=False, gqa_func="mean", mode="andpro",
                     lookahead_particle_idx=0).items():
        setattr(model.config, k, v)
        if tcfg is not model.config:
            setattr(tcfg, k, v)

    from lorekv.monkeypatch.lookahead_runner import generate_with_lookahead

    ids = torch.randint(0, 500, (1, args.tokens), device=args.device)
    mask = torch.ones_like(ids)

    class _Tok:
        eos_token_id = 0
        def encode(self, s, **k): return [0]

    try:
        if args.method == "lookahead":
            out = generate_with_lookahead(model, _Tok(), ids, mask, args.gen, [0])
        else:
            # Baselines compress inside the prefill forward and then decode normally,
            # so they exercise hybrid_baseline_hijack + the ragged cache without the
            # lookahead runner at all.
            setattr(model.config, "lookahead_phase", "final")
            if tcfg is not model.config:
                setattr(tcfg, "lookahead_phase", "final")
            gen = model.generate(ids, max_new_tokens=args.gen, do_sample=False,
                                 use_cache=True)
            out = gen[0, ids.shape[1]:]
    except Exception:
        print("\n--- FAILED ---")
        traceback.print_exc()
        return 1

    print(f"\ngenerated {out.numel()} token(s): {out.tolist()}")
    # Prove eviction really happened on the layers we own.
    for idx in comp:
        from lorekv.monkeypatch.lookahead_runner import _decoder_layers
        _lyr = _decoder_layers(model)[idx]
        kvc = getattr(getattr(_lyr, "self_attn", None), "kv_cluster", None)
        if kvc is not None and getattr(kvc, "head_lens", None) is not None:
            print(f"  layer {idx}: per-head kept = {kvc.head_lens.tolist()} "
                  f"(prefill was {args.tokens})")
    print("\nRESULT: PASS -- full cycle completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
