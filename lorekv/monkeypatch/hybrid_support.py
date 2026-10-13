"""Layer-type support for the hybrid-attention backbones (Qwen3.5, Gemma 4).

In a hybrid model only some layers keep a KV cache that grows with the context.
Qwen3.5-9B has 8 full-attention layers and 24 gated-delta-net (linear-attention)
layers. Gemma-4-12B has 8 full-attention layers and 40 sliding-window layers
(window 1024).

The other layers have a state of constant size:

    Qwen3.5-9B   The linear-attention state is a fixed recurrent state
                 (0.036 GiB in total for all such layers).
    Gemma-4-12B  A sliding-window layer keeps min(n, 1024) entries
                 (0.31 GiB in total for n >= 1024).

Share of the KV cache in the full-attention layers (bfloat16, measured):

    n            Qwen3.5-9B     Gemma-4-12B
    4,096          77.9%           16.7%
    8,192          87.6%           28.6%
    32,768         96.6%           61.5%
    64,000         98.2%           75.8%

For Gemma 4 the share uses the per-layer geometry. A full-attention layer has
1 KV head of 512 dimensions; a sliding-window layer has 8 KV heads of 256.

LORE-KV thus compresses only the full-attention layers (paper App. E.2):

1. :func:`compressible_layers` returns the indices of the layers whose cache
   grows with n.
2. :func:`resolve_budget` gives the budget of each compressible layer for a
   nominal budget B (see ``HYBRID_BUDGET_MODE``).
"""

from __future__ import annotations

import os

# Layer types whose KV cache grows with the sequence length. The other types have a
# state of fixed size or a bounded window and are not compressed.
GROWING_LAYER_TYPES = ("full_attention",)

# Layer types that are not compressed, with the reason. The run manifest records them.
BOUNDED_LAYER_TYPES = {
    "linear_attention": "fixed-size recurrent state; O(1) in context length",
    "conv": "fixed-size convolution state; O(1) in context length",
    "sliding_attention": "cache bounded by sliding_window; O(1) beyond the window",
    "chunked_attention": "cache bounded by the chunk; O(1) beyond it",
}


def layer_types(config):
    """Return the list of layer types, or None for a model with full attention in each layer."""
    text_cfg = getattr(config, "text_config", None) or config
    lt = getattr(text_cfg, "layer_types", None)
    if lt is None:
        return None
    return list(lt)


def is_hybrid(config) -> bool:
    """Return True if the model has layers other than full-attention layers."""
    lt = layer_types(config)
    return bool(lt) and any(t not in GROWING_LAYER_TYPES for t in lt)


def compressible_layers(config):
    """Return the indices of the layers whose KV cache grows with the context length.

    For a model with full attention in each layer, the function returns all layers.
    """
    lt = layer_types(config)
    if not lt:
        text_cfg = getattr(config, "text_config", None) or config
        return list(range(int(getattr(text_cfg, "num_hidden_layers", 0))))
    return [i for i, t in enumerate(lt) if t in GROWING_LAYER_TYPES]


def layer_is_compressible(config, layer_idx: int) -> bool:
    """Return True if layer ``layer_idx`` is compressible."""
    lt = layer_types(config)
    if not lt:
        return True
    if layer_idx >= len(lt):
        return False
    return lt[layer_idx] in GROWING_LAYER_TYPES


# --- Budget accounting -------------------------------------------------------
#
# If only k of L layers are compressible, a budget B has two interpretations:
#
#   per_layer (default)
#       Each compressible layer keeps B entries, as on a dense backbone. The results
#       are comparable to the dense backbones at the same B. The total retained
#       cache is k * B, not L * B.
#
#   total
#       The compressible layers share L * B entries, so each layer keeps B * L / k.
#       The total cache is equal to that of a dense model, but the budget of each
#       layer is larger (4x on Qwen3.5). The results are not comparable to the
#       dense backbones at the same B.
HYBRID_BUDGET_MODE = os.environ.get("HYBRID_BUDGET_MODE", "per_layer").lower()


def resolve_budget(config, base_capacity: int, mode: str | None = None) -> int:
    """Return the budget of each compressible layer for a nominal budget B."""
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
    """Read a config value that transformers 5.x can guard as a per-layer attribute.

    The function reads the value from ``to_dict()``. If the value is a per-layer
    list, it returns the maximum.
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
    """Return the fraction of the KV cache in the compressible layers at context length n.

    The result is 1.0 for a model with full attention in each layer. The function
    counts elements, so the data type has no effect.
    """
    lt = layer_types(config)
    if not lt:
        return 1.0
    # Use the per-layer geometry if the architecture has it. Gemma 4 full-attention
    # layers have head_dim 512 and 1 KV head; its sliding-window layers have 256 and
    # 8. The top-level config gives the sliding-window values.
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
    # An attention layer stores K and V. With attention_k_eq_v (Gemma 4) the
    # full-attention layers store only one tensor per token.
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
            # Linear-attention or convolution state: fixed size.
            lk = int(_cfg_get(config, "linear_key_head_dim", 0))
            lv = int(_cfg_get(config, "linear_value_head_dim", 0))
            nk = int(_cfg_get(config, "linear_num_key_heads", 0))
            nv = int(_cfg_get(config, "linear_num_value_heads", 0))
            ck = int(_cfg_get(config, "linear_conv_kernel_dim", 0))
            bounded += nk * lk * lk + nv * lv * lv + nk * lk * ck
    total = growing + bounded
    return (growing / total) if total else 1.0


def hybrid_manifest(config, base_capacity: int, n_tokens: int = 64000) -> dict:
    """Return a record of the compressed layers and the layers that are not compressed.

    The record also gives the reason for each layer type (paper App. E.2).
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
