"""Ministral3 text tower for transformers 4.44, expressed in the Llama idiom.

Structurally Ministral3 is Llama-3 (SwiGLU MLP, RMSNorm pre-norm blocks, GQA,
no attention bias, no sliding window). It departs in exactly two places, both
reproduced from `transformers.models.ministral3.modular_ministral3` in 5.x:

  1. RoPE uses a YaRN inverse-frequency schedule. 4.44 also ships a "yarn"
     branch, but an older one: it takes the correction range from
     `max_position_embeddings` (262144) rather than from
     `original_max_position_embeddings` (16384), and it derives the attention
     factor as 0.1*log(factor)+1 rather than from the mscale ratio. Using it
     would silently mis-rotate every position, so the schedule is recomputed
     here with the newer formula.
  2. After RoPE the query is scaled by
         1 + beta * log(1 + floor(pos / original_max_position_embeddings)),
     the "llama-4" length scaling. It is exactly 1 for every position below
     16384 and 1.069 in the 16384-32767 band, so it matters for the long
     LongBench/RULER prompts and is a no-op for the short ones.

Verified against upstream `transformers.models.ministral3` -- see
scripts/validate_models_ext.py.
"""

import math
from typing import Optional, Tuple

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
    LlamaSdpaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)

from models_ext.configuration_ministral3 import Ministral3Config

Ministral3RMSNorm = LlamaRMSNorm
Ministral3MLP = LlamaMLP


def compute_yarn_inv_freq(config: Ministral3Config, device=None):
    """YaRN inverse frequencies + attention factor, per transformers 5.x.

    Line-for-line `_compute_yarn_parameters` from transformers 5.16
    `modeling_rope_utils.py`, restricted to the keys a Ministral3 config sets.
    """
    rp = config.rope_parameters
    base = float(rp["rope_theta"])
    dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    factor = float(rp["factor"])
    original_max = int(rp["original_max_position_embeddings"])
    beta_fast = rp.get("beta_fast") or 32
    beta_slow = rp.get("beta_slow") or 1
    truncate = rp.get("truncate", True)

    attention_factor = rp.get("attention_factor")
    if attention_factor is None:
        mscale, mscale_all_dim = rp.get("mscale"), rp.get("mscale_all_dim")

        def get_mscale(scale, m=1):
            return 1.0 if scale <= 1 else 0.1 * m * math.log(scale) + 1.0

        if mscale and mscale_all_dim:
            attention_factor = float(get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim))
        else:
            attention_factor = get_mscale(factor)

    def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings, truncate):
        low = find_correction_dim(low_rot, dim, base, max_position_embeddings)
        high = find_correction_dim(high_rot, dim, base, max_position_embeddings)
        if truncate:
            low, high = math.floor(low), math.ceil(high)
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(lo, hi, n):
        if lo == hi:
            hi += 0.001  # prevent singularity
        return torch.clamp((torch.arange(n, dtype=torch.float32) - lo) / (hi - lo), 0, 1)

    pos_freqs = base ** (torch.arange(0, dim, 2).to(device=device, dtype=torch.float) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max, truncate)
    extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2).to(device=device, dtype=torch.float)
    inv_freq = (
        inv_freq_interpolation * (1 - extrapolation_factor) + inv_freq_extrapolation * extrapolation_factor
    )
    return inv_freq, attention_factor


def get_llama_4_attn_scale(position_ids: torch.Tensor, beta: float, max_position_embeddings: int) -> torch.Tensor:
    """[bsz, 1, seq, 1] query scale, broadcasting over heads and head_dim."""
    scaling = 1 + beta * torch.log(1 + torch.floor(position_ids / max_position_embeddings))
    return scaling[:, None, :, None]


class Ministral3RotaryEmbedding(nn.Module):
    """`LlamaRotaryEmbedding`'s forward with the YaRN schedule above.

    Not a subclass: LlamaRotaryEmbedding.__init__ resolves the schedule through
    `ROPE_INIT_FUNCTIONS[config.rope_scaling["rope_type"]]`, which is the 4.44
    yarn this file exists to avoid.
    """

    def __init__(self, config: Ministral3Config, device=None):
        super().__init__()
        self.config = config
        self.rope_type = "yarn"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        inv_freq, self.attention_scaling = compute_yarn_inv_freq(config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos, sin = emb.cos(), emb.sin()
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Ministral3Attention(LlamaAttention):
    """Eager attention with the YaRN rotary and the llama-4 query scale."""

    def __init__(self, config: Ministral3Config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        head_dim = getattr(config, "head_dim", None) or self.head_dim
        if head_dim != self.head_dim:
            self.head_dim = head_dim
            bias = config.attention_bias
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * head_dim, bias=bias)
            self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * head_dim, bias=bias)
            self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * head_dim, bias=bias)
            self.o_proj = nn.Linear(self.num_heads * head_dim, self.hidden_size, bias=bias)
        self.rotary_emb = Ministral3RotaryEmbedding(config)
        self.sliding_window = config.sliding_window
        self._llama4_beta = config.rope_parameters.get("llama_4_scaling_beta")
        self._llama4_window = config.rope_parameters.get("original_max_position_embeddings")

    def project_qkv(self, hidden_states):
        """q/k/v as [bsz, heads, q_len, head_dim]. Pre-RoPE; identical to Llama."""
        bsz, q_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def scale_query(self, query_states, position_ids):
        """Post-RoPE llama-4 length scaling. Exactly 1.0 below the 16384 window."""
        if not self._llama4_beta or position_ids is None:
            return query_states
        scale = get_llama_4_attn_scale(position_ids, self._llama4_beta, self._llama4_window)
        return query_states * scale.to(query_states.dtype)

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
        query_states = self.scale_query(query_states, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


class Ministral3FlashAttention2(Ministral3Attention, LlamaFlashAttention2):
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
        query_states = self.scale_query(query_states, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

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


class Ministral3SdpaAttention(Ministral3Attention, LlamaSdpaAttention):
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
            return Ministral3Attention.forward(
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
        query_states = self.scale_query(query_states, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
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


MINISTRAL3_ATTENTION_CLASSES = {
    "eager": Ministral3Attention,
    "flash_attention_2": Ministral3FlashAttention2,
    "sdpa": Ministral3SdpaAttention,
}


class Ministral3DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: Ministral3Config, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = MINISTRAL3_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        self.mlp = Ministral3MLP(config)
        self.input_layernorm = Ministral3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Ministral3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Ministral3PreTrainedModel(LlamaPreTrainedModel):
    config_class = Ministral3Config
    base_model_prefix = "model"
    _no_split_modules = ["Ministral3DecoderLayer"]


class Ministral3Model(Ministral3PreTrainedModel, LlamaModel):
    """Llama-3's model body with Ministral3 decoder layers.

    `self.rotary_emb` is the YaRN one, so the `position_embeddings` LlamaModel.forward
    computes once and threads through the layers already carries the right schedule;
    the per-layer `rotary_emb` is only a fallback for callers that pass none.
    """

    def __init__(self, config: Ministral3Config):
        Ministral3PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Ministral3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Ministral3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Ministral3RotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.post_init()


class Ministral3ForCausalLM(Ministral3PreTrainedModel, LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Ministral3Config):
        Ministral3PreTrainedModel.__init__(self, config)
        self.model = Ministral3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


__all__ = [
    "Ministral3Config",
    "Ministral3Attention",
    "Ministral3FlashAttention2",
    "Ministral3SdpaAttention",
    "Ministral3DecoderLayer",
    "Ministral3Model",
    "Ministral3ForCausalLM",
    "Ministral3PreTrainedModel",
    "compute_yarn_inv_freq",
    "get_llama_4_attn_scale",
]
