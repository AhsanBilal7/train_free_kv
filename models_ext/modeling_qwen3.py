"""Qwen3 for transformers 4.44, expressed in the Llama idiom.

Qwen3 differs from Llama-3 in exactly two ways that matter here:

  1. per-head RMSNorm on the query and key projections ("QK-norm"), applied
     after the head split and BEFORE RoPE, over the head dimension only;
  2. `head_dim` is a config field rather than `hidden_size // num_heads`
     (they coincide for Qwen3-8B: 4096 / 32 = 128, but not for every Qwen3 size).

Everything else -- SwiGLU MLP without bias, RMSNorm pre-norm blocks, GQA,
default RoPE, no attention bias -- is Llama. So each class below subclasses its
Llama counterpart and overrides only what the two differences force, which keeps
the surface the eviction hijacks patch (`Qwen3FlashAttention2.forward`,
`Qwen3Model.forward`, `Qwen3ForCausalLM.prepare_inputs_for_generation`)
identical in signature to the Llama one they were written against.

Verified against upstream `transformers.models.qwen3` -- see
scripts/validate_models_ext.py.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaFlashAttention2,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaSdpaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)

from models_ext.configuration_qwen3 import Qwen3Config

Qwen3RMSNorm = LlamaRMSNorm
Qwen3MLP = LlamaMLP
Qwen3RotaryEmbedding = LlamaRotaryEmbedding


class Qwen3Attention(LlamaAttention):
    """Eager attention with QK-norm. Base class for the sdpa/flash variants."""

    def __init__(self, config: Qwen3Config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        # LlamaAttention hardcodes head_dim = hidden_size // num_heads and sizes the
        # projections from it. Qwen3 states head_dim explicitly, so rebuild the four
        # projections whenever the two disagree (they agree for Qwen3-8B).
        head_dim = getattr(config, "head_dim", None) or self.head_dim
        if head_dim != self.head_dim:
            self.head_dim = head_dim
            bias = config.attention_bias
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * head_dim, bias=bias)
            self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * head_dim, bias=bias)
            self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * head_dim, bias=bias)
            self.o_proj = nn.Linear(self.num_heads * head_dim, self.hidden_size, bias=bias)
            self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        # Normalized over the head dimension only, so no reshape is needed after it
        # (unlike OLMo2, which norms the flattened projection).
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # `use_sliding_window` is False on every released Qwen3 dense checkpoint;
        # honour it if a future one turns it on.
        self.sliding_window = config.sliding_window

    def project_qkv(self, hidden_states):
        """q/k/v as [bsz, heads, q_len, head_dim], with QK-norm applied. Pre-RoPE."""
        bsz, q_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        return q, k, v.transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        query_states, key_states, value_states = self.project_qkv(hidden_states)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


class Qwen3FlashAttention2(Qwen3Attention, LlamaFlashAttention2):
    """Llama-3's FlashAttention-2 path with QK-norm in the projection step.

    The MRO puts Qwen3Attention first so `project_qkv`/`__init__` come from it,
    while `_flash_attn_uses_top_left_mask` and the rest of the FA2 plumbing come
    from LlamaFlashAttention2.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False
        bsz, q_len, _ = hidden_states.size()
        query_states, key_states, value_states = self.project_qkv(hidden_states)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # flash-attn wants [bsz, seq, heads, head_dim]; it repeats KV itself.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=0.0 if not self.training else self.attention_dropout,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value


class Qwen3SdpaAttention(Qwen3Attention, LlamaSdpaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # No SDPA kernel returns the weights; fall back exactly as Llama does.
            return Qwen3Attention.forward(
                self, hidden_states, attention_mask, position_ids, past_key_value,
                output_attentions, use_cache, cache_position, position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()
        query_states, key_states, value_states = self.project_qkv(hidden_states)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with non-contiguous inputs and a mask is a known issue on CUDA.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        is_causal = True if causal_mask is None and q_len > 1 else False
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value


QWEN3_ATTENTION_CLASSES = {
    "eager": Qwen3Attention,
    "flash_attention_2": Qwen3FlashAttention2,
    "sdpa": Qwen3SdpaAttention,
}


class Qwen3DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        # Deliberately skip LlamaDecoderLayer.__init__, which would build a Llama
        # attention module; everything else it does is reproduced verbatim below.
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = QWEN3_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen3PreTrainedModel(LlamaPreTrainedModel):
    config_class = Qwen3Config
    base_model_prefix = "model"
    _no_split_modules = ["Qwen3DecoderLayer"]


class Qwen3Model(Qwen3PreTrainedModel, LlamaModel):
    """Llama-3's model body with Qwen3 decoder layers.

    `LlamaModel.forward` is inherited unchanged -- and, crucially, so is whatever
    a hijack has patched onto it, which is why the eviction packages only need to
    replace the attention forward for this backbone.
    """

    def __init__(self, config: Qwen3Config):
        Qwen3PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()


class Qwen3ForCausalLM(Qwen3PreTrainedModel, LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen3Config):
        Qwen3PreTrainedModel.__init__(self, config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


__all__ = [
    "Qwen3Config",
    "Qwen3Attention",
    "Qwen3FlashAttention2",
    "Qwen3SdpaAttention",
    "Qwen3DecoderLayer",
    "Qwen3Model",
    "Qwen3ForCausalLM",
    "Qwen3PreTrainedModel",
]
