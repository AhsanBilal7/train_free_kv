"""LORE-KV attention forwards for the hybrid-attention backbones (transformers 5.x).

Qwen3.5 (``qwen3_5``) and Gemma 4 (``gemma4_unified``) exist only in transformers
5.x. Release 4.48 removed the ``*FlashAttention2`` classes that the dense patches
replace. The 5.x attention forward has this signature::

    forward(hidden_states, position_embeddings, attention_mask, past_key_values, **kw)

The model computes ``position_embeddings`` (cos, sin) one time and gives them to
each layer. The rotary embedding thus does not depend on the cache length.

Scope:
    Only the layers with a KV cache that grows with the context are compressed
    (see :mod:`.hybrid_support`).

    Qwen3.5: ``Qwen3_5DecoderLayer`` makes a ``Qwen3_5Attention`` only for
    full-attention layers. The linear-attention layers use
    ``Qwen3_5GatedDeltaNet``. A patch of ``Qwen3_5Attention`` thus applies only to
    the correct layers.

    Gemma 4: One class (``Gemma4UnifiedTextAttention``) serves both layer types.
    ``self.layer_type`` identifies the type, and the patch checks it. The
    sliding-window layers are not compressed.

Each patched forward calls the original forward for a layer that it does not
compress. An error thus gives an uncompressed result, not an incorrect one.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from lorekv.monkeypatch.scoring import init_lookahead_snapkv
from lorekv.monkeypatch.hybrid_support import (
    compressible_layers, layer_is_compressible, resolve_budget)


_VARLEN_FALLBACK_WARNED = False


def _varlen_attend(q, k, v, cu_qlen, cu_klen, max_seqlen_k, softmax_scale=None):
    """Compute attention of one query per head against that head's slice of a ragged cache.

    The function uses ``flash_attn_varlen_func`` if it is available. Otherwise it pads
    the slices to [H, max_len, D], sets the padding to -inf and uses PyTorch. The
    fallback gives the same result: with one query per head, each query sees its full
    slice and needs no causal mask.
    """
    try:
        import flash_attn
        # An import-only stub can be installed in this environment. A successful
        # import thus does not show that the kernel is available.
        flash_attn_varlen_func = (None if getattr(flash_attn, "__lorekv_stub__", False)
                                  else flash_attn.flash_attn_varlen_func)
        # flash-attn supports only CUDA. Select the path from the tensor device, so
        # that scripts/harness_hybrid_cycle.py can run on CPU.
        if not q.is_cuda:
            flash_attn_varlen_func = None
    except Exception:
        flash_attn_varlen_func = None

    if flash_attn_varlen_func is not None:
        try:
            return flash_attn_varlen_func(q, k, v, cu_qlen, cu_klen, 1, max_seqlen_k,
                                          causal=True, softmax_scale=softmax_scale)
        except RuntimeError as exc:
            # The head-dimension limit of the kernel depends on the build, the dtype
            # and the shapes. If the kernel fails, use the PyTorch path. It gives the
            # same result (difference 1.2e-07 against a per-head reference).
            if "head dimension" not in str(exc):
                raise
            global _VARLEN_FALLBACK_WARNED
            if not _VARLEN_FALLBACK_WARNED:
                print("[lorekv] flash_attn_varlen rejected head_dim=%d (%s); "
                      "using the exact torch fallback for the ragged decode."
                      % (q.shape[-1], str(exc)[:60]), flush=True)
                _VARLEN_FALLBACK_WARNED = True

    h = cu_qlen.numel() - 1
    d = q.shape[-1]
    starts = cu_klen[:-1].tolist()
    ends = cu_klen[1:].tolist()
    lens = [e - s for s, e in zip(starts, ends)]
    m = max(lens)
    kf, vf = k.view(-1, d), v.view(-1, d)
    kp = kf.new_zeros((h, m, d))
    vp = vf.new_zeros((h, m, d))
    mask = torch.zeros((h, m), dtype=torch.bool, device=q.device)
    for i, (s, ln) in enumerate(zip(starts, lens)):
        kp[i, :ln] = kf[s:s + ln]
        vp[i, :ln] = vf[s:s + ln]
        mask[i, :ln] = True
    qh = q.view(h, 1, d).float()
    # Use the scaling of the module, not 1/sqrt(d). Gemma 4 sets scaling = 1.0,
    # because q_norm contains the scale. An additional 1/sqrt(d) makes the softmax
    # almost uniform.
    scale = (d ** -0.5) if softmax_scale is None else float(softmax_scale)
    logits = torch.bmm(qh, kp.float().transpose(1, 2)) * scale        # [H,1,m]
    logits = logits.masked_fill(~mask.unsqueeze(1), float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    out = torch.bmm(probs, vp.float())                                # [H,1,d]
    return out.to(q.dtype).view(h, 1, d)


def _flat_append(flat, new_rows, head_lens, cu_klen):
    """Append one row per head to a ragged [sum_h len_h, D] buffer.

    The function uses ``tiny_api_cuda.update_flatten_view`` if it is available. The
    fallback makes the new buffer with a scatter. Each head slice grows by one row,
    so the destination offset of head h increases by h.
    """
    try:
        from tiny_api_cuda import update_flatten_view
        return update_flatten_view(flat, new_rows, head_lens, cu_klen)
    except Exception:
        pass
    h = int(head_lens.numel())
    lens = head_lens.tolist()
    starts = cu_klen[:h].tolist()
    out = flat.new_empty((flat.shape[0] + h, flat.shape[-1]))
    w = 0
    for i in range(h):
        ln = int(lens[i])
        s0 = int(starts[i])
        out[w:w + ln] = flat[s0:s0 + ln]
        out[w + ln] = new_rows[i]
        w += ln + 1
    return out


def make_hybrid_flat_cache(config, past_key_values=None):
    """Make the cache for the decode phase of a hybrid model.

    transformers 5.x builds ``DynamicCache(config=...)`` as a list of per-layer cache
    objects (DynamicLayer, DynamicSlidingWindowLayer, LinearAttentionLayer). This
    function replaces the objects of the compressed layers with a ragged per-head
    layer. The other layers keep their own implementation and state.
    """
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache(config=config)
    for idx in compressible_layers(config):
        if idx < len(cache.layers):
            cache.layers[idx] = _SplitHeadFlattenLayer()
    cache._lorekv_hybrid_flat = True
    if past_key_values is not None:
        # Keep the prefill state of the layers that are not replaced. compress()
        # fills the compressed layers.
        for idx, layer in enumerate(getattr(past_key_values, "layers", [])):
            if idx < len(cache.layers) and not isinstance(cache.layers[idx], _SplitHeadFlattenLayer):
                cache.layers[idx] = layer
    return cache


class _SplitHeadFlattenLayer:
    """Cache of one compressed layer, with all heads in one ragged buffer.

    After eviction each head keeps a different number of tokens (Eq. 8, Eq. 11). The
    keys and values are thus stored as [sum_h len_h, D] and read by
    ``flash_attn_varlen_func``. ``update`` appends one token to each head.

    The class does not subclass DynamicLayer. DynamicLayer uses
    ``self.keys.shape[-2]`` as the sequence length, which is not correct for this
    layout.
    """

    # The attributes of the CacheLayerMixin / DynamicLayer interface. The methods
    # that are not correct for a ragged layout (get_seq_length, get_mask_sizes,
    # crop, batch_*) are replaced or refused below.
    is_sliding = False
    is_initialized = True
    is_compileable = False
    supports_early_init = False
    _layer_type = None

    def __init__(self):
        self.keys = None
        self.values = None

    def update(self, key_states, value_states, cache_kwargs=None):
        if self.keys is None:
            self.keys, self.values = key_states, value_states
            return self.keys, self.values
        cache_kwargs = cache_kwargs or {}
        head_lens = cache_kwargs["head_lens"]
        cu_klen = cache_kwargs["cu_klen"]
        dim = key_states.shape[-1]
        self.keys = _flat_append(self.keys.view(-1, dim), key_states.view(-1, dim),
                                 head_lens, cu_klen)
        self.values = _flat_append(self.values.view(-1, dim), value_states.view(-1, dim),
                                   head_lens, cu_klen)
        return self.keys, self.values

    # The flattened layout has no single sequence length. Return 1, so that the
    # position advances by one token per step, as on the 4.44 path.
    def get_seq_length(self, *_a, **_k):
        return 0 if self.keys is None else 1

    def get_max_cache_shape(self, *_a, **_k):
        return None

    def get_max_length(self, *_a, **_k):
        return None

    def get_mask_sizes(self, query_length, *_a, **_k):
        """Return ``(kv_length, kv_offset)`` for mask construction, as DynamicLayer does.

        The argument is the query length (an int). The patched decode step reads the real
        per-head lengths from ``cu_klen``. This value only keeps the model-level mask code
        consistent.
        """
        return self.get_seq_length() + int(query_length), 0

    def reset(self):
        self.keys = self.values = None

    def lazy_initialization(self, *_a, **_k):
        return None

    def offload(self, *_a, **_k):
        return None

    def prefetch(self, *_a, **_k):
        return None

    def crop(self, *_a, **_k):
        # Each head keeps a different subset of positions, so a crop to the first N
        # positions is not defined. Raise an error.
        raise NotImplementedError(
            "crop() is not defined for a ragged per-head cache (post-eviction)")

    def batch_repeat_interleave(self, *_a, **_k):
        raise NotImplementedError("batch ops are not defined for a ragged per-head cache")

    def batch_select_indices(self, *_a, **_k):
        raise NotImplementedError("batch ops are not defined for a ragged per-head cache")

    def reorder_cache(self, *_a, **_k):
        raise NotImplementedError("beam reordering is not defined for a ragged per-head cache")


# Register the class as a virtual subclass of CacheLayerMixin. transformers uses
# `isinstance(layer, CacheLayerMixin)` to find the attention layers of a cache (for
# example in DynamicCache.get_seq_length). The registration passes this check and
# does not inherit the DynamicLayer methods, which are not correct for a ragged
# layout.
CacheLayerMixin = None
try:
    from transformers.cache_utils import CacheLayerMixin as _CLM
    _CLM.register(_SplitHeadFlattenLayer)
    CacheLayerMixin = _CLM
except Exception:
    pass


def make_hybrid_model_forward(orig_forward):
    """Wrap a 5.x ``TextModel.forward`` so that the decode phase uses the ragged cache.

    The wrapper replaces the cache before the decoder layers run. After
    ``compress()`` the compressed layers have no rectangular [B, H, S, D] tensor, and
    the standard DynamicCache cannot store them. The original forward does all other
    operations.
    """
    def forward(self, *args, **kwargs):
        if getattr(self.config, "lookahead_phase", "collect") == "final":
            pkv = kwargs.get("past_key_values")
            # Make the cache only one time. generate() gives the cache back at each
            # step, and a new cache would remove the eviction.
            if not getattr(pkv, "_lorekv_hybrid_flat", False):
                kwargs["past_key_values"] = make_hybrid_flat_cache(self.config, pkv)
        return orig_forward(self, *args, **kwargs)
    forward._lorekv_wrapped = True
    return forward


def _phase(self):
    return getattr(self.config, "lookahead_phase", "collect")


def _init_hybrid_snapkv(self):
    """Call ``init_lookahead_snapkv`` and add the values that the 5.x attention classes do not have.

    ``Qwen3_5Attention`` and ``Gemma4UnifiedTextAttention`` do not store
    ``num_heads``, so the function reads it from the config. It also sets
    ``base_capacity`` to the budget of each compressible layer
    (:func:`.hybrid_support.resolve_budget`).
    """
    if not hasattr(self, "kv_cluster"):
        cfg = self.config
        text_cfg = getattr(cfg, "text_config", None) or cfg
        if not hasattr(cfg, "num_hidden_layers"):
            cfg.num_hidden_layers = int(text_cfg.num_hidden_layers)
        # Resolve the budget one time, before the cluster is made, so that all
        # later steps (including Eq. 11) use it.
        if not getattr(cfg, "_lorekv_budget_resolved", False):
            cfg.base_capacity = resolve_budget(cfg, cfg.base_capacity)
            cfg._lorekv_budget_resolved = True
    init_lookahead_snapkv(self)
    if getattr(self.kv_cluster, "num_heads", None) is None:
        text_cfg = getattr(self.config, "text_config", None) or self.config
        self.kv_cluster.num_heads = int(text_cfg.num_attention_heads)
    # Use num_key_value_groups of the attention module, not of the global config.
    # Gemma 4 sets num_key_value_heads per layer, so the global value can be
    # different from the value of this layer.
    self.kv_cluster.num_key_value_groups = int(self.num_key_value_groups)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def _collect_or_lookahead(self, query_states, key_states, value_states, past_key_values):
    """Record the data that ``compress()`` needs, and return the full K and V.

    The prefill (``collect``) and the continuations (``lookahead``) call this
    function. No eviction occurs here. The continuations need the full cache, so
    :mod:`.lookahead_runner` calls ``compress()`` after all M continuations.
    """
    k_compact, v_compact = past_key_values.update(key_states, value_states, self.layer_idx)
    if _phase(self) == "collect":
        self.kv_cluster.capture_window_queries(query_states)
        # Store K and V with the KV-head count. compress() repeats them to the
        # query-head count for one layer at a time, which saves memory.
        self.kv_cluster.store_full_kv(k_compact, v_compact)
    else:
        self.kv_cluster.capture_lookahead_query(
            getattr(self.config, "lookahead_particle_idx", 0), query_states)
    return k_compact, v_compact


def _final_phase_varlen(self, query_states, key_states, value_states, past_key_values):
    """Decode one token against the ragged per-head cache after eviction.

    Each head of a compressed layer keeps a different number of tokens, so the cache
    is flattened and read with the varlen kernel, as on the dense backbones.
    """
    cache_kwargs = {
        "head_lens": self.kv_cluster.head_lens,
        "cu_klen": self.kv_cluster.cu_klen,
    }
    n_rep = self.num_key_value_groups
    key_states = _repeat_kv(key_states, n_rep)
    value_states = _repeat_kv(value_states, n_rep)
    key_states, value_states = past_key_values.update(
        key_states, value_states, self.layer_idx, cache_kwargs)

    n_heads = self.kv_cluster.num_heads
    self.kv_cluster.klen_sum += n_heads
    self.kv_cluster.max_seqlen_k += 1
    self.kv_cluster.cu_klen += self.kv_cluster.cu_offset
    self.kv_cluster.head_lens += 1

    head_dim = self.head_dim
    q = query_states.view(-1, 1, head_dim)
    k = key_states.view(-1, 1, head_dim)
    v = value_states.view(-1, 1, head_dim)

    return _varlen_attend(q, k, v, self.kv_cluster.cu_qlen, self.kv_cluster.cu_klen,
                          self.kv_cluster.max_seqlen_k,
                          softmax_scale=getattr(self, "scaling", None))


# --------------------------------------------------------------------------
# Qwen3.5  (model_type "qwen3_5")
# --------------------------------------------------------------------------
def causal_lookahead_qwen3_5_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values=None,
    **kwargs,
):
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        apply_rotary_pos_emb, eager_attention_forward)
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    # Qwen3_5Attention is used only in full-attention layers. Check this condition.
    if not layer_is_compressible(self.config, self.layer_idx):
        return self._lorekv_orig_forward(
            hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)

    _init_hybrid_snapkv(self)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # Qwen3.5 gates the attention output. q_proj gives 2 * head_dim per head, and the
    # second half is a sigmoid gate that is applied after attention. Both phases keep
    # the gate.
    query_states, gate = torch.chunk(
        self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)

    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if _phase(self) in ("collect", "lookahead"):
        k_compact, v_compact = _collect_or_lookahead(
            self, query_states, key_states, value_states, past_key_values)
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward)
        attn_output, _ = attention_interface(
            # Do not repeat K and V here. The 5.x attention interface applies repeat_kv.
            self, query_states, k_compact, v_compact,
            attention_mask, dropout=0.0, scaling=self.scaling, **kwargs)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    else:
        out = _final_phase_varlen(
            self, query_states, key_states, value_states, past_key_values)
        bsz, q_len = input_shape
        attn_output = out.reshape(bsz, self.kv_cluster.num_heads, q_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)

    attn_output = attn_output * torch.sigmoid(gate)
    return self.o_proj(attn_output), None


# --------------------------------------------------------------------------
# Gemma 4  (model_type "gemma4_unified")
# --------------------------------------------------------------------------
def causal_lookahead_gemma4_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    shared_kv_states: Optional[dict] = None,
    past_key_values=None,
    **kwargs,
):
    from transformers.models.gemma4_unified.modeling_gemma4_unified import (
        apply_rotary_pos_emb, eager_attention_forward)
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    # Gemma 4 uses one class for sliding-window and full-attention layers. Compress
    # only the full-attention layers. A layer with is_kv_shared_layer uses the K and V
    # of another layer, so it is not compressed.
    if not layer_is_compressible(self.config, self.layer_idx) or getattr(
            self, "is_kv_shared_layer", False):
        return self._lorekv_orig_forward(
            hidden_states, position_embeddings, attention_mask, shared_kv_states,
            past_key_values, **kwargs)

    _init_hybrid_snapkv(self)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    cos, sin = position_embeddings

    query_states = self.q_proj(hidden_states).view(hidden_shape)
    query_states = self.q_norm(query_states)
    query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2).transpose(1, 2)

    # attention_k_eq_v: v_proj is None, and the values are the keys before the norm.
    key_states = self.k_proj(hidden_states).view(hidden_shape)
    value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states
    key_states = self.k_norm(key_states)
    key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2).transpose(1, 2)
    value_states = self.v_norm(value_states).transpose(1, 2)

    if _phase(self) in ("collect", "lookahead"):
        k_compact, v_compact = _collect_or_lookahead(
            self, query_states, key_states, value_states, past_key_values)
        if getattr(self, "store_full_length_kv", False) and shared_kv_states is not None:
            shared_kv_states[self.layer_type] = (k_compact, v_compact)
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward)
        attn_output, _ = attention_interface(
            # Do not repeat K and V here. The 5.x attention interface applies repeat_kv.
            self, query_states, k_compact, v_compact,
            attention_mask, dropout=0.0, scaling=self.scaling,
            # The original Gemma 4 forward also gives this argument. It is None on
            # the full-attention layers.
            sliding_window=getattr(self, "sliding_window", None), **kwargs)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    else:
        out = _final_phase_varlen(
            self, query_states, key_states, value_states, past_key_values)
        bsz, q_len = input_shape
        attn_output = out.reshape(bsz, self.kv_cluster.num_heads, q_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)

    return self.o_proj(attn_output), None
