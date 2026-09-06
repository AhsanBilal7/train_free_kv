"""causal_lookahead_ext_flash_attn2_forward -- causal_lookahead_snapkv's hijack for the models_ext backbones (Ministral3).

GENERATED from causal_lookahead_snapkv/monkeypatch/causal_llama_hijack.py by scripts/gen_ext_hijacks.py;
do not edit by hand. Re-run the generator after any change to the Llama hijack
so the two stay in lockstep. See that script's docstring for what is edited and
why.
"""

from typing import List, Optional, Tuple, Union

import torch
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.utils import logging, is_flash_attn_2_available

from causal_lookahead_snapkv.monkeypatch.snapkv_utils import init_lookahead_snapkv

logger = logging.get_logger(__name__)

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func


def causal_lookahead_ext_flash_attn2_forward(
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
    init_lookahead_snapkv(self)
    if isinstance(past_key_value, StaticCache):
        raise ValueError(
            "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
            "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
        )

    output_attentions = False
    bsz, q_len, _ = hidden_states.size()

    # The one place this backbone differs from Llama: it sizes its heads from
    # config.head_dim. `project_qkv` is defined on the model's own attention class
    # (models_ext), so this single forward serves every backbone registered there.
    query_states, key_states, value_states = self.project_qkv(hidden_states)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    # Ministral3 scales the query by 1 + beta*log(1 + floor(pos/16384)) after RoPE;
    # backbones without that scaling do not define the hook and are unaffected.
    if hasattr(self, "scale_query"):
        query_states = self.scale_query(query_states, position_ids)

    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

    phase = getattr(self.config, "lookahead_phase", "collect")

    if phase in ("collect", "lookahead"):
        key_states_compact, value_states_compact = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if phase == "collect":
            self.kv_cluster.capture_window_queries(query_states)
            # Store the COMPACT (num_key_value_heads-count) K/V, not the GQA-repeated one --
            # repeating here would persist a much larger tensor on every layer for the entire
            # lookahead rollout. compress() repeats transiently, one layer at a time, instead.
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
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=True,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    else:
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

        attn_output = flash_attn_varlen_func(query_states, key_states, value_states, cu_seqlens_q, cu_seqlens_k,
                                              max_seqlen_q, max_seqlen_k, causal=True)
        assert bsz == 1
        attn_output = attn_output.reshape(bsz, self.num_heads, q_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)
    attn_weights = None
    return attn_output, attn_weights, past_key_value
