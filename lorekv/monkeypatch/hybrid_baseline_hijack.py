"""AnDPro / CriticalKV on hybrid-attention backbones (transformers 5.x).

The baselines are structurally simpler than LORE-KV: there is no lookahead
rollout, so there are only two branches.

    prefill (q_len > 1)  score with the prompt's own queries, evict immediately
                         via kv_cluster.update_kv(), then run attention on the
                         FULL (pre-eviction) states for this forward pass
    decode  (q_len == 1) serve the ragged post-eviction cache with varlen

Both packages (`andpro_snapkv`, `adaptive_snapkv`) expose the same shape --
`init_*_snapkv(self)` then `self.kv_cluster.update_kv(k, q, v)` -- so one
parameterised builder covers both rather than two near-identical files that can
drift apart.

Layer scoping, the per-layer flat cache and the fall-through-when-not-ours rule
are shared with the LORE-KV hijack; see causal_lookahead_hybrid_hijack.py and
hybrid_support.py for why only the full-attention layers are compressed (paper App. E.2).
"""

from __future__ import annotations

from typing import Callable

import torch

from lorekv.monkeypatch.causal_lookahead_hybrid_hijack import (
    _repeat_kv, _varlen_attend, make_hybrid_flat_cache)
from lorekv.monkeypatch.hybrid_support import (
    layer_is_compressible, resolve_budget)


def _allow_uniform_per_layer_reads(cfg):
    """Let the 4.44 baseline code read per-layer-guarded attributes, when it is safe.

    Gemma 4's config lists `num_key_value_heads` and `head_dim` in
    `per_layer_attributes`, so a plain read raises
    AmbiguousGlobalPerLayerAttributeError -- even when every layer holds the same
    value, which IS the case for the 12B checkpoint. andpro/adaptive's
    init_*_snapkv computes `num_attention_heads // num_key_value_heads` that way.

    transformers offers `allow_global_per_layer_attribute_access` for exactly this,
    with the caveat that the caller must know the config is homogeneous. So verify
    it rather than assert it: read every layer's value and only lift the guard when
    they genuinely all agree. On a checkpoint that really does vary per layer this
    stays closed and the error surfaces, which is the correct outcome -- the
    baseline's single global `num_key_value_groups` would be wrong there.
    """
    for c in {id(x): x for x in (cfg, getattr(cfg, "text_config", None)) if x is not None}.values():
        guarded = getattr(c, "per_layer_attributes", None)
        if not guarded:
            continue
        try:
            raw = c.to_dict()
        except Exception:
            continue
        uniform = True
        for name in guarded:
            v = raw.get(name)
            if isinstance(v, (list, tuple)) and len(set(v)) > 1:
                uniform = False
                break
        if uniform:
            try:
                c.allow_global_per_layer_attribute_access = True
            except Exception:
                pass


def _init_baseline(self, init_fn):
    """init_*_snapkv plus what the 5.x attention classes do not carry."""
    if not hasattr(self, "kv_cluster"):
        cfg = self.config
        text_cfg = getattr(cfg, "text_config", None) or cfg
        if not hasattr(cfg, "num_hidden_layers"):
            cfg.num_hidden_layers = int(text_cfg.num_hidden_layers)
        _allow_uniform_per_layer_reads(cfg)
        if not getattr(cfg, "_lorekv_budget_resolved", False):
            cfg.base_capacity = resolve_budget(cfg, cfg.base_capacity)
            cfg._lorekv_budget_resolved = True
    init_fn(self)
    if getattr(self.kv_cluster, "num_heads", None) is None:
        text_cfg = getattr(self.config, "text_config", None) or self.config
        self.kv_cluster.num_heads = int(text_cfg.num_attention_heads)
    # The ATTENTION MODULE's value is authoritative, not the config's. init_*_snapkv
    # computes num_key_value_groups as a global
    # `config.num_attention_heads // config.num_key_value_heads`, but Gemma 4 keeps
    # num_key_value_heads in per_layer_config, so the global read can disagree with
    # what this layer's projections actually emit. The cluster then repeat_kv's by the
    # wrong factor and the score matmul fails ("size of tensor a (4) must match tensor
    # b (2)"). self.num_key_value_groups was derived from this layer's own config when
    # the module was built, so it always matches the tensors.
    self.kv_cluster.num_key_value_groups = int(self.num_key_value_groups)


def _decode_varlen(self, query_states, key_states, value_states, past_key_values):
    cache_kwargs = {"head_lens": self.kv_cluster.head_lens,
                    "cu_klen": self.kv_cluster.cu_klen}
    key_states = _repeat_kv(key_states, self.num_key_value_groups)
    value_states = _repeat_kv(value_states, self.num_key_value_groups)
    key_states, value_states = past_key_values.update(
        key_states, value_states, self.layer_idx, cache_kwargs)

    self.kv_cluster.klen_sum += self.kv_cluster.num_heads
    self.kv_cluster.max_seqlen_k += 1
    self.kv_cluster.cu_klen += self.kv_cluster.cu_offset
    self.kv_cluster.head_lens += 1

    d = self.head_dim
    return _varlen_attend(
        query_states.view(-1, 1, d), key_states.view(-1, 1, d), value_states.view(-1, 1, d),
        self.kv_cluster.cu_qlen, self.kv_cluster.cu_klen, self.kv_cluster.max_seqlen_k,
        softmax_scale=getattr(self, "scaling", None))


def make_qwen3_5_baseline_forward(init_fn):
    """Build a Qwen3.5 attention forward for a baseline package."""

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, **kwargs):
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            apply_rotary_pos_emb, eager_attention_forward)
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        if not layer_is_compressible(self.config, self.layer_idx):
            return self._lorekv_orig_forward(
                hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)

        _init_baseline(self, init_fn)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q_len = input_shape[-1]

        # Output gate -- see causal_lookahead_hybrid_hijack for why it is carried.
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if q_len != 1:
            k_c, v_c = self.kv_cluster.update_kv(key_states, query_states, value_states)
            past_key_values.update(k_c, v_c, self.layer_idx)
            attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, eager_attention_forward)
            attn_output, _ = attention_interface(
                # NOT pre-repeated -- the interface applies repeat_kv itself.
                self, query_states, key_states, value_states,
                attention_mask, dropout=0.0, scaling=self.scaling, **kwargs)
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        else:
            out = _decode_varlen(self, query_states, key_states, value_states, past_key_values)
            bsz = input_shape[0]
            attn_output = out.reshape(bsz, self.kv_cluster.num_heads, q_len, self.head_dim)
            attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)

        return self.o_proj(attn_output * torch.sigmoid(gate)), None

    return forward


def make_gemma4_baseline_forward(init_fn):
    """Build a Gemma 4 attention forward for a baseline package."""

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                shared_kv_states=None, past_key_values=None, **kwargs):
        from transformers.models.gemma4_unified.modeling_gemma4_unified import (
            apply_rotary_pos_emb, eager_attention_forward)
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        if not layer_is_compressible(self.config, self.layer_idx) or getattr(
                self, "is_kv_shared_layer", False):
            return self._lorekv_orig_forward(
                hidden_states, position_embeddings, attention_mask, shared_kv_states,
                past_key_values, **kwargs)

        _init_baseline(self, init_fn)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q_len = input_shape[-1]
        cos, sin = position_embeddings

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        # attention_k_eq_v: v_proj is None and the value stream IS the key stream.
        value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states
        key_states = self.k_norm(key_states)
        key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2).transpose(1, 2)
        value_states = self.v_norm(value_states).transpose(1, 2)

        if q_len != 1:
            k_c, v_c = self.kv_cluster.update_kv(key_states, query_states, value_states)
            past_key_values.update(k_c, v_c, self.layer_idx)
            if getattr(self, "store_full_length_kv", False) and shared_kv_states is not None:
                shared_kv_states[self.layer_type] = (key_states, value_states)
            attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, eager_attention_forward)
            attn_output, _ = attention_interface(
                # NOT pre-repeated -- the interface applies repeat_kv itself.
                self, query_states, key_states, value_states,
                attention_mask, dropout=0.0, scaling=self.scaling,
                sliding_window=getattr(self, "sliding_window", None), **kwargs)
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        else:
            out = _decode_varlen(self, query_states, key_states, value_states, past_key_values)
            bsz = input_shape[0]
            attn_output = out.reshape(bsz, self.kv_cluster.num_heads, q_len, self.head_dim)
            attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)

        return self.o_proj(attn_output), None

    return forward


def _make_baseline_model_forward(orig_forward):
    """Install the ragged cache for a BASELINE run.

    LORE-KV has three phases and only needs the flat cache in the "final" one, so
    make_hybrid_model_forward gates on lookahead_phase == "final". A baseline has no
    phases at all: it evicts inside the prefill forward and decodes from the evicted
    cache on the very next token. Reusing the gated wrapper meant the swap never
    happened -- update_kv wrote a flattened 2-D tensor into a stock DynamicLayer,
    and the first decode step died in transformers' own cat():
        RuntimeError: Tensors must have same number of dimensions: got 2 and 4
    So install it unconditionally here, still only for the compressible layers.
    """
    def forward(self, *args, **kwargs):
        pkv = kwargs.get("past_key_values")
        if pkv is not None and not getattr(pkv, "_lorekv_hybrid_flat", False):
            kwargs["past_key_values"] = make_hybrid_flat_cache(self.config, pkv)
        return orig_forward(self, *args, **kwargs)
    forward._lorekv_wrapped = True
    return forward


def _patch(attn_cls, model_cls, fwd):
    if not hasattr(attn_cls, "_lorekv_orig_forward"):
        attn_cls._lorekv_orig_forward = attn_cls.forward
    attn_cls.forward = fwd
    if not getattr(model_cls.forward, "_lorekv_wrapped", False):
        model_cls.forward = _make_baseline_model_forward(model_cls.forward)


def replace_hybrid_baseline(which: str):
    """Patch Qwen3.5 + Gemma 4 for a baseline. `which` is 'andpro' or 'adaptive'.

    A baseline decodes against the ragged cache from the first generated token, so
    there is no "final" phase flag to switch on -- the model forward marks the
    cache as flat up front by setting lookahead_phase="final".
    """
    if which == "andpro":
        from andpro_snapkv.monkeypatch.snapkv_utils import init_andpro_snapkv as init_fn
    elif which == "adaptive":
        from adaptive_snapkv.monkeypatch.snapkv_utils import init_adaptive_snapkv as init_fn
    else:
        raise ValueError(f"unknown baseline {which!r}")

    for mod_name, attn_name, model_name, builder in (
        ("qwen3_5", "Qwen3_5Attention", "Qwen3_5TextModel", make_qwen3_5_baseline_forward),
        ("gemma4_unified", "Gemma4UnifiedTextAttention", "Gemma4UnifiedTextModel",
         make_gemma4_baseline_forward),
    ):
        try:
            M = __import__(f"transformers.models.{mod_name}.modeling_{mod_name}",
                           fromlist=[attn_name])
            _patch(getattr(M, attn_name), getattr(M, model_name), builder(init_fn))
        except (ImportError, ModuleNotFoundError, AttributeError):
            continue
