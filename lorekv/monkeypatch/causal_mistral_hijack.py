"""LORE-KV attention and model forwards for Mistral (transformers 4.44.2).

``causal_lookahead_MistralModel_forward`` replaces ``MistralModel.forward``.
``causal_lookahead_mistral_flash_attn2_forward`` replaces
``MistralFlashAttention2.forward``. Both are adapted from the transformers 4.44.2
implementation (Apache-2.0) and from the varlen decode step of the AdaKV and
AnDPro patches. ``config.lookahead_phase`` selects the operation (see
:mod:`lorekv.monkeypatch.lookahead_runner`):

* ``collect``: Causal flash attention on a standard DynamicCache. The layer
  records Q_W and the full K and V on its ``kv_cluster``.
* ``lookahead``: The same attention for one step of a continuation. The layer
  records the query of the step.
* ``final``: One decode step on the evicted per-head cache with
  ``flash_attn_varlen_func``.

The model forward returns the hidden state of the last position only. The runner
uses only these logits, and the prefill does not compute the [n, vocab] product.
"""
from typing import List, Optional, Tuple, Union

import torch
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.models.mistral.modeling_mistral import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.utils import logging, is_flash_attn_2_available
from lorekv.monkeypatch.scoring import init_lookahead_snapkv, DynamicCacheSplitHeadFlatten

logger = logging.get_logger(__name__)

if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func


# The phase is read from `self.config`, not from a layer's kv_cluster, because the
# attention modules create their kv_cluster lazily on the first call.
def causal_lookahead_MistralModel_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    phase = getattr(self.config, "lookahead_phase", "collect")
    if phase == "final":
        return_legacy_cache = True
        past_key_values = DynamicCacheSplitHeadFlatten.from_legacy_cache(past_key_values)
    else:
        return_legacy_cache = False
        if past_key_values is None:
            past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, use_cache, output_attentions
    )

    hidden_states = inputs_embeds
    next_decoder_cache = None

    for decoder_layer in self.layers:
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = layer_outputs[0]
        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

    hidden_states = self.norm(hidden_states)

    next_cache = next_decoder_cache if use_cache else None
    if return_legacy_cache:
        next_cache = next_cache.to_legacy_cache()

    # Only the last position's logits are ever consumed by lookahead_runner (prefill, lookahead
    # decode steps, and final decode steps are all only scored/sampled at the last position) --
    # truncating here avoids materializing the full [n, vocab] lm_head matmul during prefill.
    hidden_states = hidden_states[:, -1, :].unsqueeze(1)

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, None, None] if v is not None)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
    )


def causal_lookahead_mistral_flash_attn2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
):
    if isinstance(past_key_value, StaticCache):
        raise ValueError(
            "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
            "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
        )
    init_lookahead_snapkv(self)

    output_attentions = False
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    cache_kwargs = {"sin": sin, "cos": cos}

    phase = getattr(self.config, "lookahead_phase", "collect")

    if phase in ("collect", "lookahead"):
        # Plain causal attention against a stock (un-flattened, un-evicted) DynamicCache --
        # eviction is deferred to lookahead_runner.generate_with_lookahead's explicit compress()
        # call, which happens only after all M lookahead particles have been rolled out.
        key_states_compact, value_states_compact = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if phase == "collect":
            self.kv_cluster.capture_window_queries(query_states)
            # Store the COMPACT (num_key_value_heads-count) K/V, not the GQA-repeated one --
            # repeating here would persist a 4x-larger (for Mistral) tensor on every layer for the
            # entire lookahead rollout. compress() repeats transiently, one layer at a time, instead.
            self.kv_cluster.store_full_kv(key_states_compact, value_states_compact)
        else:
            particle_idx = getattr(self.config, "lookahead_particle_idx", 0)
            self.kv_cluster.capture_lookahead_query(particle_idx, query_states)

        key_states = repeat_kv(key_states_compact, self.num_key_value_groups)
        value_states = repeat_kv(value_states_compact, self.num_key_value_groups)
        q_t = query_states.transpose(1, 2)
        k_t = key_states.transpose(1, 2)
        v_t = value_states.transpose(1, 2)
        attn_output = _flash_attention_forward(
            q_t, k_t, v_t,
            attention_mask, q_len,
            position_ids=position_ids,
            dropout=0.0,
            sliding_window=getattr(self.config, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=True,
        )
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim).contiguous()
    else:
        # phase == "final": decode against the evicted per-head cache (as in AnDPro / Ada-KV).
        cache_kwargs["head_lens"] = self.kv_cluster.head_lens
        cache_kwargs["cu_klen"] = self.kv_cluster.cu_klen
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        self.kv_cluster.klen_sum += self.num_heads
        self.kv_cluster.max_seqlen_k += 1
        self.kv_cluster.cu_klen += self.kv_cluster.cu_offset
        self.kv_cluster.head_lens += 1

        query_states = query_states.view(-1, 1, self.head_dim)
        key_states = key_states.view(-1, 1, self.head_dim)
        value_states = value_states.view(-1, 1, self.head_dim)

        cu_seqlens_q = self.kv_cluster.cu_qlen
        cu_seqlens_k = self.kv_cluster.cu_klen
        max_seqlen_q = 1
        max_seqlen_k = self.kv_cluster.max_seqlen_k

        attn_output = flash_attn_varlen_func(query_states, key_states, value_states, cu_seqlens_q,
                                              cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=True)
        assert bsz == 1
        attn_output = attn_output.reshape(bsz, self.num_heads, q_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)
    attn_weights = None
    return attn_output, attn_weights, past_key_value


def prepare_inputs_for_generation_lookahead_mistral(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_position=None,
    position_ids=None,
    use_cache=True,
    **kwargs,
):
    if past_key_values is not None:
        if inputs_embeds is not None:
            input_ids = input_ids[:, -cache_position.shape[0]:]
        elif input_ids.shape[1] != cache_position.shape[0]:
            input_ids = input_ids[:, cache_position]

    if attention_mask is not None and position_ids is None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1]:]
            position_ids = position_ids.clone(memory_format=torch.contiguous_format)

    if inputs_embeds is not None and cache_position[0] == 0:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids.contiguous()}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
        }
    )
    return model_inputs
