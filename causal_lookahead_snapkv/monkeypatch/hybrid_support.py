"""Layer-type awareness for hybrid-attention backbones (Qwen3.5, Gemma 4).

The 2026 generation of open decoders is hybrid: only a minority of layers keep a
KV cache that grows with context. Qwen3.5-9B is 8 full-attention layers and 24
gated-delta-net (linear attention) layers; Gemma 4 12B is 8 full-attention and 40
sliding-window (1024) layers.

Read as "LORE-KV only covers 8/32 layers" that sounds fatal. It is the wrong
denominator. What matters for a cache-eviction method is not how many layers it
touches but how much of the *cache* it governs, and the non-full layers are O(1)
in context length by construction:

    Qwen3.5-9B   linear-attention state is a fixed recurrent state (conv kernel +
                 linear K/V heads), 0.036 GiB TOTAL, independent of n.
    Gemma 4 12B  sliding layers hold min(n, 1024) entries, so 0.31 GiB total once
                 n >= 1024 and constant thereafter.

Only the full-attention layers grow. Measured share of the KV cache held by the
full-attention layers (bf16):

    n            Qwen3.5-9B     Gemma 4 12B
    4,096          77.9%           16.7%
    8,192          87.6%           28.6%
    32,768         96.6%           61.5%
    64,000         98.2%           75.8%

Gemma 4's numbers come from its PER-LAYER geometry: full-attention layers are
1 KV head x 512 dims (512 elements/token) while sliding layers are 8 x 256 (2048),
so a sliding layer holds 4x more per token than a compressed one. The top-level
config reports the sliding values, and reading it globally overstated the
compressed share as 92.6% at 64k.

So LORE-KV applied only to the full-attention layers still governs 98% (Qwen3.5)
and 76% (Gemma 4) of the KV cache at 64k. The non-full layers are not a gap in coverage -- they are
already O(1) in context length, which is the problem KV eviction exists to solve.
In the long-context regime this work targets, those layers have already solved it
by construction.

That is the framing this module implements: **LORE-KV composes with hybrid
attention -- the architecture bounds what it can, and LORE-KV compresses what
remains, which is where the memory actually is.**

Two mechanisms carry it:

1. `compressible_layers(config)` returns exactly the layer indices whose cache
   grows with n. For Qwen3.5 this is free -- `Qwen3_5DecoderLayer` only builds a
   `Qwen3_5Attention` for full-attention layers, so patching that class already
   scopes us correctly. Gemma 4 uses ONE attention class for both layer types
   (distinguished by `self.layer_type`), so there the check must be explicit.
   Getting this wrong on Gemma 4 would silently compress sliding layers, which is
   not the experiment: evicting a 1024-entry window to B=128 and evicting a 64k
   cache to B=128 are different operations and must not be averaged together.

2. `resolve_budget()` decides what "budget B" means when only a subset of layers
   is compressible. See HYBRID_BUDGET_MODE below -- this is a reporting decision,
   not an implementation detail, and it has to be stated in the paper.
"""

from __future__ import annotations

import os

# Layer types whose KV cache grows with sequence length. Everything else is either
# a fixed-size recurrent state or a bounded window, and is left alone.
GROWING_LAYER_TYPES = ("full_attention",)

# Layer types we deliberately do NOT compress, with the reason. Used for the
# run manifest so a results directory records what was skipped and why.
BOUNDED_LAYER_TYPES = {
    "linear_attention": "fixed-size recurrent state; O(1) in context length",
    "conv": "fixed-size convolution state; O(1) in context length",
    "sliding_attention": "cache bounded by sliding_window; O(1) beyond the window",
    "chunked_attention": "cache bounded by the chunk; O(1) beyond it",
}


def layer_types(config):
    """The per-layer type list, or None for a uniform (all-full-attention) model."""
    text_cfg = getattr(config, "text_config", None) or config
    lt = getattr(text_cfg, "layer_types", None)
    if lt is None:
        return None
    return list(lt)


def is_hybrid(config) -> bool:
    lt = layer_types(config)
    return bool(lt) and any(t not in GROWING_LAYER_TYPES for t in lt)


def compressible_layers(config):
    """Indices of layers whose KV cache grows with n -- the ones LORE-KV acts on.

    A uniform model returns every layer, so this is safe to call unconditionally
    and callers need no is_hybrid() branch.
    """
    lt = layer_types(config)
    if not lt:
        text_cfg = getattr(config, "text_config", None) or config
        return list(range(int(getattr(text_cfg, "num_hidden_layers", 0))))
    return [i for i, t in enumerate(lt) if t in GROWING_LAYER_TYPES]


def layer_is_compressible(config, layer_idx: int) -> bool:
    lt = layer_types(config)
    if not lt:
        return True
    if layer_idx >= len(lt):
        return False
    return lt[layer_idx] in GROWING_LAYER_TYPES


# --- Budget accounting -------------------------------------------------------
#
# With only k of L layers compressible, "budget B" is ambiguous, and the two
# readings are different experiments:
#
#   per_layer (default)
#       Every compressible layer keeps B entries, exactly as on a dense backbone.
#       A row is then directly comparable to the Mistral/Llama/Qwen2.5 rows at the
#       same B -- same per-layer working set, same eviction pressure per layer.
#       Total retained cache is k*B rather than L*B, so the *overall* retention
#       percentage is lower than a dense model at the same B, and must be reported
#       as such rather than quietly compared.
#
#   total
#       The compressible layers share the budget a dense model of the same depth
#       would have spent (L*B), i.e. each gets B*L/k. This holds total cache
#       footprint constant against a dense baseline, at the cost of much weaker
#       per-layer pressure -- on Qwen3.5 that is 4x the per-layer budget, which
#       makes the task substantially easier and is NOT comparable to a dense row
#       at the same nominal B.
#
# Default is per_layer because it keeps the per-layer semantics of every existing
# table; `total` exists so the memory-matched comparison can be run deliberately.
HYBRID_BUDGET_MODE = os.environ.get("HYBRID_BUDGET_MODE", "per_layer").lower()


def resolve_budget(config, base_capacity: int, mode: str | None = None) -> int:
    """Per-compressible-layer budget, given the nominal B and the accounting mode."""
    mode = (mode or HYBRID_BUDGET_MODE).lower()
    if mode not in ("per_layer", "total"):
        raise ValueError(f"HYBRID_BUDGET_MODE must be per_layer|total, got {mode!r}")
    if mode == "per_layer":
        return int(base_capacity)
    lt = layer_types(config)
    if not lt:
        return int(base_capacity)
    k = len(compressible_layers(config))
    if k == 0:
        return int(base_capacity)
    return int(round(base_capacity * len(lt) / k))


def _cfg_get(config, key, default=0):
    """Read a config value that transformers 5.x may guard as per-layer.

    Gemma 4's text config lists `num_key_value_heads` and `head_dim` in
    `per_layer_attributes`, so a plain getattr raises
    AmbiguousGlobalPerLayerAttributeError even when every layer holds the same
    value (which is the case for the 12B checkpoint). to_dict() returns what the
    config.json actually says. When the value really does vary per layer it comes
    back as a list and we take the max, which keeps this an upper bound on the
    bounded share -- i.e. it never flatters the compressed share.
    """
    text_cfg = getattr(config, "text_config", None) or config
    try:
        v = getattr(text_cfg, key)
    except Exception:
        v = None
    if v is None:
        try:
            v = text_cfg.to_dict().get(key, default)
        except Exception:
            v = default
    if isinstance(v, (list, tuple)):
        v = max(v) if v else default
    return v if v is not None else default


def cache_share_of_growing_layers(config, n_tokens: int) -> float:
    """Fraction of the KV cache held by compressible layers at context length n.

    This is the number that justifies the whole approach, so it is computed rather
    than asserted, and written into each run's manifest. Returns 1.0 for a uniform
    model. Counts elements, not bytes -- dtype cancels in the ratio.
    """
    lt = layer_types(config)
    if not lt:
        return 1.0
    # Per-LAYER geometry where the architecture has it. Gemma 4's full-attention
    # layers are head_dim=512 / kv_heads=1 while its sliding layers are 256 / 8 --
    # the top-level config reports the sliding values, so reading globally
    # understates the compressed layers' cache by 512*1 vs 256*8 = 4x in the wrong
    # direction. The whole justification for this port is a cache-share number, so
    # it has to come from the layers actually being measured.
    per_layer = None
    try:
        text_cfg = getattr(config, "text_config", None) or config
        per_layer = getattr(text_cfg, "per_layer_config", None)
    except Exception:
        per_layer = None

    def _geom(i):
        if per_layer is not None:
            try:
                e = per_layer[i]
                return int(e.num_key_value_heads), int(e.head_dim)
            except Exception:
                pass
        return (int(_cfg_get(config, "num_key_value_heads", 0)),
                int(_cfg_get(config, "head_dim", 0)))

    kv_heads = int(_cfg_get(config, "num_key_value_heads", 0))
    head_dim = int(_cfg_get(config, "head_dim", 0))
    window = _cfg_get(config, "sliding_window", None)
    # An attention layer caches K *and* V -- except when the architecture ties them.
    # Gemma 4 sets attention_k_eq_v, and its full-attention layers have v_proj=None
    # and reuse the key tensor, so they cache one tensor per token, not two.
    kv_mult = 1 if bool(_cfg_get(config, "attention_k_eq_v", False)) else 2
    per_tok = max(kv_mult * kv_heads * head_dim, 1)

    growing = bounded = 0
    for i, t in enumerate(lt):
        kvh, hd = _geom(i)
        per_tok_i = max(kv_mult * kvh * hd, 1)
        if t in GROWING_LAYER_TYPES:
            growing += n_tokens * per_tok_i
        elif t in ("sliding_attention", "chunked_attention") and window:
            bounded += min(n_tokens, int(window)) * per_tok_i
        else:
            # Linear/conv state: fixed, and small relative to any real context.
            lk = int(_cfg_get(config, "linear_key_head_dim", 0))
            lv = int(_cfg_get(config, "linear_value_head_dim", 0))
            nk = int(_cfg_get(config, "linear_num_key_heads", 0))
            nv = int(_cfg_get(config, "linear_num_value_heads", 0))
            ck = int(_cfg_get(config, "linear_conv_kernel_dim", 0))
            bounded += nk * lk * lk + nv * lv * lv + nk * lk * ck
    total = growing + bounded
    return (growing / total) if total else 1.0


def hybrid_manifest(config, base_capacity: int, n_tokens: int = 64000) -> dict:
    """What was compressed, what was not, and why -- written next to the results.

    A reviewer's first question about a hybrid-backbone row is "did you actually
    compress this model or just a slice of it?". This answers it from the config
    rather than from prose.
    """
    lt = layer_types(config)
    comp = compressible_layers(config)
    skipped = {}
    if lt:
        for i, t in enumerate(lt):
            if t not in GROWING_LAYER_TYPES:
                skipped.setdefault(t, {"count": 0, "reason": BOUNDED_LAYER_TYPES.get(t, "not cache-growing")})
                skipped[t]["count"] += 1
    return {
        "hybrid": is_hybrid(config),
        "num_layers": len(lt) if lt else len(comp),
        "compressed_layers": comp,
        "num_compressed_layers": len(comp),
        "skipped_layer_types": skipped,
        "budget_mode": HYBRID_BUDGET_MODE,
        "nominal_budget": int(base_capacity),
        "per_layer_budget": resolve_budget(config, base_capacity),
        "cache_share_of_compressed_layers_at_n": {
            str(n): round(cache_share_of_growing_layers(config, n), 4)
            for n in (4096, 8192, 32768, int(n_tokens))
        },
    }
