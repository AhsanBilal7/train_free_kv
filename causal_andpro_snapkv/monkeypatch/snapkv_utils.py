import warnings

import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math
from typing import List, Optional, Tuple, Union, Any,Dict
from transformers.cache_utils import Cache, DynamicCache
from flash_attn import flash_attn_func
# perform qk calculation and get indices
# this version will not update in inference mode

class DynamicCacheSplitHeadFlatten(Cache):
    """
    Flattened version of DynamicCacheSplitHead
    """
    def __init__(self) ->None:
        # Token wise List[]  Head wise KV List[torch.Tensor]
        super().__init__()
        self.key_cache: List[List[torch.Tensor]] = []
        self.value_cache: List[List[torch.Tensor]] = []
        self._seen_tokens = 0

    def __len__(self):
        return len(self.key_cache)

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))

    def __getitem__(self, layer_idx: int) -> Tuple[Tuple[torch.Tensor],Tuple[torch.Tensor]]:
        if layer_idx < len(self):
            return (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # NOTE: k, v = [head_num](bs, 1, seqlen, dim)
        # each layer is a flatten layout like:
        # [head_0_len + head_1_len + ..., dim]
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            assert self.key_cache[layer_idx].dim() == 2
            bs, head, seqlen, dim = key_states.shape
            assert bs == 1 and seqlen == 1
            # NOTE: phase 2. we got [bs, head, seqlen, dim] as k, v input
            head_lens = cache_kwargs["head_lens"]
            cu_klen = cache_kwargs["cu_klen"]

            # TODO: wrap as a python interface
            from tiny_api_cuda import update_flatten_view
            new_key_cache = update_flatten_view(self.key_cache[layer_idx].view(-1,dim), key_states.view(-1, dim), head_lens, cu_klen)
            new_value_cache = update_flatten_view(self.value_cache[layer_idx].view(-1,dim), value_states.view(-1, dim), head_lens, cu_klen)


            self.key_cache[layer_idx] = new_key_cache
            self.value_cache[layer_idx] = new_value_cache


        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0

        # TODO: return 1 to means has content for now
        return 1
        # return max(map(lambda states: states.shape[-2], self.key_cache[layer_idx]))

    def get_max_length(self) -> Optional[int]:
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCacheEachHead":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache



# class DynamicCacheSplitHead(Cache):
#     """
#     demo for illustrate the splited cache update
#     This class is slower than DynamicCacheSplitHeadFlatten, due to the frequent tensor copy
#     """
#     def __init__(self) ->None:
#         # Token wise List[]  Head wise KV List[torch.Tensor]
#         super().__init__()
#         self.key_cache: List[List[torch.Tensor]] = []
#         self.value_cache: List[List[torch.Tensor]] = []
#         self._seen_tokens = 0

#     def __len__(self):
#         return len(self.key_cache)

#     def __iter__(self):
#         for layer_idx in range(len(self)):
#             yield (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))

#     def __getitem__(self, layer_idx: int) -> Tuple[Tuple[torch.Tensor],Tuple[torch.Tensor]]:
#         if layer_idx < len(self):
#             return (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))
#         else:
#             raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

#     def update(
#         self,
#         key_states: List[torch.Tensor],
#         value_states: List[torch.Tensor],
#         layer_idx: int,
#         cache_kwargs: Optional[Dict[str, Any]] = None,
#     ) -> Tuple[Tuple[torch.Tensor,...],Tuple[torch.Tensor,...]]:
#         if layer_idx == 0:
#             self._seen_tokens += max(map(lambda states: states.shape[-2], key_states))

#         if len(self.key_cache)<=layer_idx:
#             self.key_cache.append(list(key_states))
#             self.value_cache.append(list(value_states))
#         else:
#             # tensor shape[ [bsz, seq, dim] * head_nums]
#             # [bsz,\sum seq,dim]
#             # [bsz,\sum seq+headnum,dim ]
#             for head_idx in range(len(key_states)):
#                 self.key_cache[layer_idx][head_idx] = torch.cat([self.key_cache[layer_idx][head_idx],key_states[head_idx]], dim=-2)
#                 self.value_cache[layer_idx][head_idx] = torch.cat([self.value_cache[layer_idx][head_idx],value_states[head_idx]], dim=-2)
#         return tuple(self.key_cache[layer_idx]), tuple(self.value_cache[layer_idx])

#     def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
#         if len(self.key_cache) <= layer_idx:
#             return 0
#         return max(map(lambda states: states.shape[-2], self.key_cache[layer_idx]))

#     def get_max_length(self) -> Optional[int]:
#         return None


#     # Tuple[Tuple[Tuple[torch.Tensor,...],Tuple[torch.Tensor,...]],...]
#     def to_legacy_cache(self)-> Tuple[Tuple[Tuple[torch.Tensor,...],Tuple[torch.Tensor,...]],...]:
#         """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
#         legacy_cache = ()
#         for layer_idx in range(len(self)):
#             legacy_cache += ((tuple(self.key_cache[layer_idx]), tuple(self.value_cache[layer_idx])),)
#         return legacy_cache
#     @classmethod
#     def from_legacy_cache(cls,past_key_values:Optional[ Tuple[Tuple[Tuple[torch.Tensor,...],Tuple[torch.Tensor,...]],...]]=None)->"DynamicCacheEachHead":
#         cache = cls()
#         if past_key_values is not None:
#             for layer_idx in range(len(past_key_values)):
#                 key_states,value_states = past_key_values[layer_idx]
#                 cache.update(list(key_states),list(value_states),layer_idx)
#         return cache



# Copied from transformers.models.llama.modeling_llama.repeat_kv for gqa_support
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class SnapKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', layer_idx = None, num_hidden_layers = None, pyram_mode = False, pyram_beta = 20,gqa_support=False,num_key_value_groups = 1, gqa_func=None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

        self.pyram_init = False
        self.pyram_mode = pyram_mode
        self.pyram_beta = pyram_beta
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        
        # support gqa
        self.gqa_support = gqa_support
        self.num_key_value_groups = num_key_value_groups
        self.gqa_func = gqa_func
        if self.gqa_support:
            assert gqa_func is not None, "gqa_func should not be None"
            assert gqa_func in ['max','mean'], "currently gqa_func should be in ['max','mean']"
            if self.num_key_value_groups == 1:
                warnings.warn("gqa_support is enabled, but num_key_value_groups is 1, which means the model is not using gqa. Please check the model configuration.")



    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(self, origin_key_states, query_states, origin_value_states):
        
        # support gqa
        key_states = repeat_kv(origin_key_states, self.num_key_value_groups)
        value_states = repeat_kv(origin_value_states, self.num_key_value_groups)
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        # compute pyramidal capacity
        if self.pyram_mode and not self.pyram_init:
            # NOTE: (max_num + min_num) / 2 == base_capacity to restrict the total capacity
            base_capacity = self.max_capacity_prompt - self.window_size
            min_num = base_capacity // self.pyram_beta
            max_num = base_capacity * 2 - min_num
                
            # if the max_num is larger than the query length, we need to adjust the max_num
            if max_num >= q_len - self.window_size:
                max_num = q_len - self.window_size
                min_num = base_capacity * 2 - max_num
        
            # NOTE: compute interval
            steps = (max_num - min_num) // (self.num_hidden_layers - 1)

            self.max_capacity_prompt = max_num - self.layer_idx * steps + self.window_size
            self.pyram_init = True
            print(f"Pyram mode adaptive capacity, layer: {self.layer_idx}, max_capacity_prompt: {self.max_capacity_prompt}, base_capacity: {self.max_capacity_prompt - self.window_size}", flush=True)

        if q_len < self.max_capacity_prompt:
            # support gqa
            if self.gqa_support:
                return origin_key_states, origin_value_states
            else:
                return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_mean = attn_weights[:, :, -self.window_size:, : -self.window_size].mean(dim = -2)
            
            # gqa_support 
            if self.gqa_support:
                attn_weights_mean = attn_weights_mean.view(attn_weights_mean.shape[0], -1, self.num_key_value_groups, attn_weights_mean.shape[-1])
                if self.gqa_func == 'max':
                    attn_weights_mean = attn_weights_mean.max(dim=-2).values
                elif self.gqa_func == 'mean':
                    attn_weights_mean = attn_weights_mean.mean(dim=-2)
                else:
                    raise ValueError('gqa_func not supported')
                
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_mean, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_mean, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')

            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            
            # support gqa
            if self.gqa_support:
                k_past_compress = origin_key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
                v_past_compress = origin_value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
                k_cur = origin_key_states[:, :, -self.window_size:, :]
                v_cur = origin_value_states[:, :, -self.window_size:, :]
            else:
                k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -self.window_size:, :]
                v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states
   

class AnDProAdaptiveSnapKVCluster():
    def __init__(self, window_size = 32, kernel_size = 7, pooling = 'maxpool',base_capacity=None,floor_alpha = None,skip = None,normalize=None,
                 layer_idx = None, num_hidden_layers = None, pyram_mode = False, pyram_beta = 20,gqa_support=False,num_key_value_groups = 1, gqa_func=None,
                 chunk_size = 4, mode = None, lambda_red = 0.0):
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.base_capacity = base_capacity #- window_size
        self.floor_ratio = floor_alpha
        self.floor_capacity = int(self.base_capacity * self.floor_ratio)
        self.adaptive_capacity = self.base_capacity - self.floor_capacity
        self.skip = skip
        
        self.chunk_size = chunk_size
        self.mode = mode

        # NOTE: Training-Free Causal AnDPro (counterfactual KV deletion scoring).
        # causal_eps mirrors the epsilon used throughout kv_wip.pdf (Eq. 10, 11, 25).
        self.causal_eps = 1e-6

        # NOTE: redundancy penalty weight for the greedy redundancy-aware chunk selection
        # (S_final(i) = S_causal(i) - lambda_red * max_{j in S} Red(i,j)). lambda_red == 0
        # disables the penalty and falls back to the original plain top-K selection.
        self.lambda_red = lambda_red

        self.normalize = normalize
        self.pyram_init = False
        self.pyram_mode = pyram_mode
        self.pyram_beta = pyram_beta
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers

        # NOTE: layer-wise meta-data
        self.head_lens = None
        self.max_seqlen_k = 0
        self.klen_sum = 0
        self.cu_klen = 0
        self.cu_offset = None
        self.cu_headlens = None

         # support gqa
        self.gqa_support = gqa_support
        self.num_key_value_groups = num_key_value_groups
        self.gqa_func = gqa_func
        if self.gqa_support:
            assert gqa_func is not None, "gqa_func should not be None"
            assert gqa_func in ['max','mean'], "currently gqa_func should be in ['max','mean']"
            if self.num_key_value_groups == 1:
                warnings.warn("gqa_support is enabled, but num_key_value_groups is 1, which means the model is not using gqa. Please check the model configuration.")


    def calcul_attn_sore(self, key_states, query_states):
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(
            head_dim)
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min,
                          device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # attn_weights_mean = attn_weights[:, :, -self.window_size:, : -self.window_size].mean(dim=-2)

        if self.gqa_support:
            attn_weights_mean = attn_weights_mean.view(attn_weights_mean.shape[0],num_heads//self.num_key_value_groups,self.num_key_value_groups,-1)
            if self.gqa_func == 'max':
                attn_weights_mean = attn_weights_mean.max(dim=-2).values
            elif self.gqa_func == 'mean':
                attn_weights_mean = attn_weights_mean.mean(dim=-2)
            else:
                raise ValueError('gqa_func not supported')

        if self.pooling == 'avgpool':
            attn_weights = F.avg_pool1d(attn_weights.squeeze(0), kernel_size=self.kernel_size,
                                        padding=self.kernel_size // 2,
                                        stride=1).unsqueeze(0)
        elif self.pooling == 'maxpool':
            attn_weights = F.max_pool1d(attn_weights.squeeze(0), kernel_size=self.kernel_size,
                                        padding=self.kernel_size // 2,
                                        stride=1).unsqueeze(0)
        else:
            raise ValueError('Pooling method not supported')
        return attn_weights

    # ------------------------------------------------------------------
    # Training-Free Causal AnDPro (counterfactual KV deletion eviction)
    #
    # Implements the scoring pipeline from kv_wip.pdf:
    #   - calcul_raw_attn_score:        a^{l,h}_{t,i}            (Eq. 2)
    #   - calcul_y_full:                y^{l,h}_t                (Eq. 3)
    #   - calcul_causal_token_score:    s^{l,h}_{t,i} aggregated  (Eq. 10/14 + Eq. 23, max over W)
    #   - select_causal_chunks:         Eq. 25 normalization + Eq. 26 top-K, at chunk granularity
    # ------------------------------------------------------------------

    def calcul_raw_attn_score(self, key_states, query_states):
        """Exact (unpooled) causal softmax attention weights a^{l,h}_{t,i} for the
        observation-window queries against the full key sequence. Unlike
        calcul_attn_sore, no smoothing pool is applied: the closed-form deletion
        effect (Eq. 10) needs the *exact* attention weight, not a neighborhood-smoothed one."""
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(
            head_dim)
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min,
                          device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]
        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        return attn_weights  # [1, num_heads, window_size, q_len], float32, no grad needed

    @staticmethod
    def calcul_y_full(attn_weights_raw, value_states):
        """Eq. 3: y_t = sum_i a_{t,i} v_i, computed over the *full* key/value sequence
        (not just the candidate/droppable region, unlike the legacy projection score)."""
        return torch.matmul(attn_weights_raw, value_states.float())  # [1, H, W, D]

    def calcul_causal_token_score(self, attn_weights_raw, value_states, y_full, direction):
        """Eq. 10/13 (closed-form deletion effect) folded into Eq. 14 (forward-only score),
        then Eq. 23 (max aggregation over the observation window). `direction` is the
        normalized anchor y_bar_t, shape [1, H, W, D]."""
        value_states_f = value_states.float()
        dir_dot_v = torch.matmul(direction, value_states_f.transpose(-1, -2))  # [1,H,W,L] : dir_t . v_j
        dir_dot_y = (direction * y_full).sum(dim=-1, keepdim=True)  # [1,H,W,1] : dir_t . y_t
        proj = dir_dot_v - dir_dot_y  # [1,H,W,L] : dir_t . (v_j - y_t)
        delta_factor = attn_weights_raw / (1.0 - attn_weights_raw + self.causal_eps)  # a_{t,j}/(1-a_{t,j}+eps)
        score = (delta_factor * proj).abs()  # [1,H,W,L] : s_{t,j}
        return score.amax(dim=2).squeeze(0)  # [H, L] : S^{l,h}_i = max_{t in W} s_{t,i}  (Eq. 23)

    def select_causal_chunks(self, token_score, q_len, num_heads, key_states=None, value_states=None):
        """Eq. 25 (per-head normalization) + Eq. 26 (budgeted top-K selection), applied at
        chunk granularity (chunk_size tokens per unit) to stay compatible with the
        flattened-cache layout used elsewhere in this codebase. Returns a boolean
        [num_heads, q_len_chunk] preserve mask, matching the legacy preserve_indices contract.

        When self.lambda_red > 0 (and key_states/value_states are supplied), selection
        switches from plain top-K to a redundancy-aware greedy rule that penalizes chunks
        that look like ones already kept:
            S_final(i) = S_causal(i) - lambda_red * max_{j in S} Red(i, j)
            Red(i, j)  = 0.5 * cos(k_i, k_j) + 0.5 * cos(v_i, v_j)
        where S is the growing set of preserved chunks (forced anchor/window chunks plus
        chunks picked so far) and k_i/v_i are each chunk's mean-pooled key/value vectors.
        """
        device = token_score.device
        pad_len = (self.chunk_size - (q_len % self.chunk_size)) % self.chunk_size
        score_padded = F.pad(token_score, (0, pad_len), value=float('-inf'))
        q_len_chunk = (q_len + pad_len) // self.chunk_size
        chunk_score = score_padded.view(num_heads, q_len_chunk, self.chunk_size).amax(dim=-1)

        drop_indices = torch.ones((num_heads, q_len_chunk), dtype=torch.bool, device=device)
        drop_indices[:, 0] = False
        drop_indices[:, - self.window_size // self.chunk_size:] = False
        num_preserved = num_heads * (self.window_size // self.chunk_size + 1)

        # Eq. 25: normalize by the per-head mean score over the candidate (droppable) region.
        # NOTE: use torch.where rather than chunk_score * drop_indices -- the padded slots hold
        # -inf, and -inf * 0 is NaN, not 0.
        zeros = torch.zeros_like(chunk_score)
        masked_for_mean = torch.where(drop_indices, chunk_score, zeros)
        mean_per_head = masked_for_mean.sum(dim=-1) / drop_indices.sum(dim=-1).clamp(min=1)
        normalized_score = chunk_score / (mean_per_head.unsqueeze(-1) + self.causal_eps)
        normalized_score = normalized_score.masked_fill(~drop_indices, float('-inf'))

        num_to_preserve = num_heads * self.base_capacity // self.chunk_size - num_preserved

        if self.lambda_red > 0 and key_states is not None and value_states is not None:
            preserve_indices = self._select_redundancy_aware(
                normalized_score, drop_indices, num_to_preserve, q_len, q_len_chunk,
                key_states, value_states)
        else:
            indices_to_preserve = torch.topk(normalized_score.reshape(-1), k=num_to_preserve, dim=-1).indices
            drop_indices = drop_indices.clone()
            drop_indices.view(-1)[indices_to_preserve] = False
            preserve_indices = ~drop_indices

        assert preserve_indices.sum().item() == num_heads * self.base_capacity // self.chunk_size
        return preserve_indices

    def _select_redundancy_aware(self, normalized_score, drop_indices, num_to_preserve,
                                  q_len, q_len_chunk, key_states, value_states):
        """Greedy selection of `num_to_preserve` chunks per head, picking one chunk at a
        time via argmax_i [S_causal(i) - lambda_red * max_{j in S} Red(i,j)] and growing S
        by the picked chunk before the next iteration. Falls back to identical results as
        plain top-K when lambda_red == 0 (the penalty term vanishes), since greedily taking
        the single highest-scoring remaining chunk each step is equivalent to top-K."""
        num_heads = normalized_score.shape[0]
        device = normalized_score.device
        head_dim = key_states.shape[-1]
        pad_len = q_len_chunk * self.chunk_size - q_len

        k = key_states.float().reshape(num_heads, q_len, head_dim)
        v = value_states.float().reshape(num_heads, q_len, head_dim)
        if pad_len > 0:
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
        # Per-chunk representative k/v vector: mean over the chunk's tokens.
        k_repr = k.view(num_heads, q_len_chunk, self.chunk_size, head_dim).mean(dim=2)
        v_repr = v.view(num_heads, q_len_chunk, self.chunk_size, head_dim).mean(dim=2)
        k_unit = k_repr / (k_repr.norm(dim=-1, keepdim=True) + self.causal_eps)
        v_unit = v_repr / (v_repr.norm(dim=-1, keepdim=True) + self.causal_eps)

        drop_indices = drop_indices.clone()
        preserved_mask = ~drop_indices  # forced-preserved chunks (anchor + recency window)

        # max_{j in S} Red(i,j), maintained as a running max so each greedy step only needs
        # to score the newly-added chunk against all candidates (not the whole set S).
        max_red = torch.full((num_heads, q_len_chunk), float('-inf'), device=device)
        for head_idx in range(num_heads):
            sel = torch.where(preserved_mask[head_idx])[0]
            if sel.numel() == 0:
                continue
            red = 0.5 * (k_unit[head_idx] @ k_unit[head_idx, sel].T) + \
                  0.5 * (v_unit[head_idx] @ v_unit[head_idx, sel].T)
            max_red[head_idx] = red.amax(dim=-1)

        # NOTE: num_to_preserve is a single *global* budget shared across all heads (it
        # mirrors the original flattened torch.topk(normalized_score.reshape(-1), ...)),
        # not a per-head budget. So each iteration picks exactly one (head, chunk) winner
        # across the whole [num_heads, q_len_chunk] grid -- never one per head per step.
        # Redundancy itself stays scoped to the winning head, since k/v live in a
        # per-head space and chunks from different heads aren't comparable.
        eligible = drop_indices.clone()
        for _ in range(num_to_preserve):
            candidate_score = normalized_score - self.lambda_red * max_red
            candidate_score = candidate_score.masked_fill(~eligible, float('-inf'))
            idx_flat = candidate_score.reshape(-1).argmax()  # i* = argmax_{(h,i)} [S_causal - lambda_red*Penalty]
            head_idx = (idx_flat // q_len_chunk).item()
            chunk_idx = (idx_flat % q_len_chunk).item()

            drop_indices[head_idx, chunk_idx] = False
            eligible[head_idx, chunk_idx] = False

            k_pick = k_unit[head_idx, chunk_idx]  # [head_dim]
            v_pick = v_unit[head_idx, chunk_idx]
            red_new = 0.5 * (k_unit[head_idx] @ k_pick) + 0.5 * (v_unit[head_idx] @ v_pick)  # Red(i, newly-picked j)
            max_red[head_idx] = torch.maximum(max_red[head_idx], red_new)

        return ~drop_indices

    def update_kv(self, origin_key_states, query_states, origin_value_states):
        if self.gqa_support:
            return self.update_kv_gqa(origin_key_states, query_states, origin_value_states)
        else:
            return self.update_kv_wo_gqa(origin_key_states, query_states, origin_value_states)


    # update kv with gqa_support
    #NOTE: this code implementation does not support gqa yet
    def update_kv_gqa(self, origin_key_states, query_states, origin_value_states):
        key_states = repeat_kv(origin_key_states, self.num_key_value_groups)
        # value_states = repeat_kv(origin_value_states, self.num_key_value_groups)

        # check if prefix phase        assert key_states.shape[-2] == query_states.shape[-2]
        _device = key_states.device
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_score= self.calcul_attn_sore(key_states,query_states)
        origin_heads_key_states = torch.split(origin_key_states, 1, dim=1)
        origin_heads_value_states = torch.split(origin_value_states, 1, dim=1)

        # compute pyramidal capacity
        if self.pyram_mode and not self.pyram_init:
            # NOTE: (max_num + min_num) / 2 == base_capacity to restrict the total capacity
            min_num = self.base_capacity // self.pyram_beta
            max_num = self.base_capacity * 2 - min_num
                
            # if the max_num is larger than the query length, we need to adjust the max_num
            if max_num >= q_len - self.window_size:
                max_num = q_len - self.window_size
                min_num = self.base_capacity * 2 - max_num
        
            # NOTE: compute interval
            steps = (max_num - min_num) // (self.num_hidden_layers - 1)

            # renew adaptive capacity
            self.base_capacity = max_num - self.layer_idx * steps
            self.floor_capacity = int(self.base_capacity * self.floor_ratio)
            self.adaptive_capacity = self.base_capacity - self.floor_capacity
            self.pyram_init = True
            print(f"Pyram mode adaptive capacity, layer: {self.layer_idx}, acap: {self.adaptive_capacity}, bcap: {self.base_capacity}, fcap: {self.floor_capacity}",  flush=True)

        def init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k):
            # init metadata
            self.head_lens = torch.tensor(k_lens, dtype=torch.int32, device=_device)
            self.klen_sum = klen_sum
            self.max_seqlen_k = max_seqlen_k
            self.cu_headlens = torch.cumsum(self.head_lens, dim=0, dtype=torch.int32)
            # init varlen flash attention metadata
            self.cu_klen = self.cu_headlens - self.head_lens
            self.cu_klen = torch.cat(
                [self.cu_klen, torch.tensor([self.klen_sum], dtype=torch.int32, device=_device)], dim=0)
            # check bug
            self.layer_qlens = torch.ones(num_heads//self.num_key_value_groups, dtype=torch.int32,device=_device)
            self.qlen_sum = num_heads//self.num_key_value_groups
            self.cu_qlen = torch.cumsum(self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
            self.cu_qlen = torch.cat(
                [self.cu_qlen, torch.tensor([self.qlen_sum], dtype=torch.int32, device=_device)], dim=0)
            
            
            if self.gqa_support:
                self.cu_offset = torch.arange(0, num_heads//self.num_key_value_groups + 1, dtype=torch.int32, device=_device)
                self.cu_head_offset = torch.arange(1, num_heads//self.num_key_value_groups +1, dtype=torch.int32, device=_device)

            else:
                self.cu_offset = torch.arange(0, num_heads + 1, dtype=torch.int32, device=_device)
                self.cu_head_offset = torch.arange(1, num_heads+1, dtype=torch.int32, device=_device)


        if self.base_capacity > attn_score.size(-1):
            init_metadata(num_heads, [q_len] * (num_heads//self.num_key_value_groups), q_len * (num_heads//self.num_key_value_groups), q_len)
            # not compress
            return origin_key_states.reshape(-1, head_dim), origin_value_states.reshape(-1, head_dim)


        sorted_attn_score,sorted_attn_score_indices = attn_score.sort(dim=-1,descending=True)
        if self.layer_idx >= self.skip:
            adaptive_attn_score = sorted_attn_score
            length = adaptive_attn_score.size(dim=-1)
            if self.normalize:
                ratio_weight = sorted_attn_score[...,:self.base_capacity].sum(dim=-1,keepdim=True)/sorted_attn_score.sum(dim=-1,keepdim=True)
                adaptive_attn_score = adaptive_attn_score*ratio_weight
            adaptive_attn_score = adaptive_attn_score.reshape(bsz,length*num_heads//self.num_key_value_groups)
            sorted_indices = torch.topk(adaptive_attn_score,k=num_heads*self.base_capacity//self.num_key_value_groups,dim=-1).indices
            sorted_indices = sorted_indices//length

            # floor_alpha capacity set
            head_adaptive_capacity = torch.zeros((bsz,num_heads//self.num_key_value_groups),device=_device,dtype = sorted_indices.dtype)
            head_adaptive_capacity.scatter_add_(-1,sorted_indices,torch.ones_like(sorted_indices,dtype=head_adaptive_capacity.dtype),)
            assert head_adaptive_capacity.sum().item() == num_heads*self.base_capacity//self.num_key_value_groups
            head_adaptive_capacity = torch.round(head_adaptive_capacity * (1-self.floor_ratio) + self.floor_capacity).int()
        else:
            head_adaptive_capacity = torch.ones((bsz,num_heads),device=_device,dtype = sorted_attn_score_indices.dtype) * self.base_capacity
        sorted_attn_score_indices = sorted_attn_score_indices.split(1,dim=1)

        heads_key_states = []
        heads_value_states = []
        assert bsz == 1
        # per head

        # reinit varlen metadata
        k_lens = []
        klen_sum = 0
        max_seqlen_k = 0
        self.cu_klen = 0


        for head_idx in range(num_heads//self.num_key_value_groups):
            cache_index = sorted_attn_score_indices[head_idx][...,:head_adaptive_capacity[0][head_idx]]

            l = cache_index.shape[-1] + self.window_size
            k_lens.append(l)
            max_seqlen_k = max(max_seqlen_k, l)
            klen_sum += l

            cache_index = cache_index.view(1, 1, -1, 1).expand(-1, -1, -1, head_dim)
            top_Kcache = origin_heads_key_states[head_idx].gather(dim=2,index=cache_index)
            top_Vcache = origin_heads_value_states[head_idx].gather(dim=2,index=cache_index)
            selected_k = torch.cat([top_Kcache,origin_heads_key_states[head_idx][:, :, -self.window_size:, :]],dim=2)
            selected_v = torch.cat([top_Vcache,origin_heads_value_states[head_idx][:, :, -self.window_size:, :]],dim=2)

            # NOTE: flatten view
            heads_key_states.append(selected_k.view(-1, head_dim))
            heads_value_states.append(selected_v.view(-1, head_dim))

        init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k)

        # NOTE: compose as flatten view
        heads_key_states = torch.cat(heads_key_states, dim=0)
        heads_value_states = torch.cat(heads_value_states, dim=0)

        return heads_key_states, heads_value_states

    
        
    # update without gqa_support
    def update_kv_wo_gqa(self,  origin_key_states, query_states, origin_value_states):
        key_states = repeat_kv(origin_key_states, self.num_key_value_groups)
        value_states = repeat_kv(origin_value_states, self.num_key_value_groups)

        # check if prefix phase        assert key_states.shape[-2] == query_states.shape[-2]
        _device = key_states.device
        bsz, num_heads, q_len, head_dim = query_states.shape
        if self.mode == "causal_forward_only":
            attn_weights_raw = self.calcul_raw_attn_score(key_states, query_states)
            attn_weights = attn_weights_raw
        else:
            attn_weights= self.calcul_attn_sore(key_states,query_states)
        origin_heads_key_states = torch.split(key_states, 1, dim=1)
        origin_heads_value_states = torch.split(value_states, 1, dim=1)

        # compute pyramidal capacity
        if self.pyram_mode and not self.pyram_init:
            # NOTE: (max_num + min_num) / 2 == base_capacity to restrict the total capacity
            min_num = self.base_capacity // self.pyram_beta
            max_num = self.base_capacity * 2 - min_num
                
            # if the max_num is larger than the query length, we need to adjust the max_num
            if max_num >= q_len - self.window_size:
                max_num = q_len - self.window_size
                min_num = self.base_capacity * 2 - max_num
        
            # NOTE: compute interval
            steps = (max_num - min_num) // (self.num_hidden_layers - 1)

            # renew adaptive capacity
            self.base_capacity = max_num - self.layer_idx * steps
            self.floor_capacity = int(self.base_capacity * self.floor_ratio)
            self.adaptive_capacity = self.base_capacity - self.floor_capacity
            self.pyram_init = True
            print(f"Pyram mode adaptive capacity, layer: {self.layer_idx}, acap: {self.adaptive_capacity}, bcap: {self.base_capacity}, fcap: {self.floor_capacity}",  flush=True)

        def init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k):
            # init metadata
            self.head_lens = torch.tensor(k_lens, dtype=torch.int32, device=_device)
            self.klen_sum = klen_sum
            self.max_seqlen_k = max_seqlen_k
            self.cu_headlens = torch.cumsum(self.head_lens, dim=0, dtype=torch.int32)
            # init varlen flash attention metadata
            self.cu_klen = self.cu_headlens - self.head_lens
            self.cu_klen = torch.cat(
                [self.cu_klen, torch.tensor([self.klen_sum], dtype=torch.int32, device=_device)], dim=0)
            self.layer_qlens = torch.ones(num_heads, dtype=torch.int32,device=_device)
            self.qlen_sum = num_heads
            self.cu_qlen = torch.cumsum(self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
            self.cu_qlen = torch.cat(
                [self.cu_qlen, torch.tensor([self.qlen_sum], dtype=torch.int32, device=_device)], dim=0)
            self.cu_offset = torch.arange(0, num_heads + 1, dtype=torch.int32, device=_device)
            self.cu_head_offset = torch.arange(1, num_heads+1, dtype=torch.int32, device=_device)

        if self.base_capacity > attn_weights.size(-1):
            init_metadata(num_heads, [q_len] * num_heads, q_len * num_heads, q_len)
            # not compress
            return key_states.reshape(-1, head_dim), value_states.reshape(-1, head_dim)

        # if you need to weight the attn_score
        pass
        # sorted_attn_score,sorted_attn_score_indices = attn_score.sort(dim=-1,descending=True)
        # NOTE: here is the core code of AnDPro (legacy chunk-projection score), and its
        # Training-Free Causal AnDPro counterfactual-deletion replacement.
        if self.layer_idx >= self.skip and self.mode == "causal_forward_only":
            # Causal forward-only score: Eq. 14 (closed-form counterfactual deletion effect,
            # anchored on the normalized full attention output y_bar_t) aggregated via Eq. 23
            # and selected via Eq. 25-26, at chunk granularity.
            y_full = self.calcul_y_full(attn_weights_raw, value_states)
            anchor = y_full / (y_full.norm(dim=-1, keepdim=True) + self.causal_eps)
            token_score = self.calcul_causal_token_score(attn_weights_raw, value_states, y_full, anchor)
            preserve_indices = self.select_causal_chunks(token_score, q_len, num_heads,
                                                           key_states=key_states, value_states=value_states)
        elif self.layer_idx >= self.skip:
            pad_len = (self.chunk_size - (q_len % self.chunk_size)) % self.chunk_size
            attn_weights = F.pad(attn_weights, (0, pad_len))
            value_states = F.pad(value_states, (0, 0, 0, pad_len))
            q_len_chunk = (q_len + pad_len) // self.chunk_size

            prods_with_dropvalue_window_mean = torch.zeros((num_heads, q_len_chunk), device=value_states.device)
            drop_indices = torch.ones((num_heads, q_len_chunk), dtype=torch.bool, device=value_states.device)
            drop_indices[:, 0] = False
            drop_indices[:, - self.window_size // self.chunk_size:] = False
            num_preserved = num_heads * (self.window_size // self.chunk_size + 1)
            for i in range(self.window_size):
                weight = attn_weights[:, :, i, :].unsqueeze(-1) # weight: [1, num_heads, q_len, 1]
                value_states_weighted = value_states * weight # value_states_weighted: [1, num_heads, q_len, head_dim]

                value_states_chunk = value_states_weighted.view(num_heads, -1, self.chunk_size, head_dim).sum(dim=-2) # value_states_chunk: [num_heads, q_len//chunk_size, chunk_size, head_dim]
                num_heads, q_len_chunk, head_dim = value_states_chunk.shape

                mask = drop_indices.unsqueeze(-1).expand(-1, -1, head_dim)
                droped_value = (value_states_chunk * mask).sum(dim=-2)
                prods_with_dropvalue_window_mean = prods_with_dropvalue_window_mean + (value_states_chunk * droped_value.unsqueeze(-2)).sum(dim=-1)

            proj_score = prods_with_dropvalue_window_mean
            proj_score[~ drop_indices] = - torch.inf
            num_to_preserve = num_heads * self.base_capacity // self.chunk_size - num_preserved
            indices_to_preserve = torch.topk(proj_score.reshape(-1), k=num_to_preserve, dim=-1).indices
            drop_indices.view(-1)[indices_to_preserve] = False
            num_preserved += num_to_preserve
            preserve_indices = ~ drop_indices
            assert num_preserved == preserve_indices.sum().item()
            assert num_preserved == num_heads * self.base_capacity // self.chunk_size

        else:
            pad_len = (self.chunk_size - (q_len % self.chunk_size)) % self.chunk_size
            q_len_chunk = (q_len + pad_len) // self.chunk_size
            preserve_indices = torch.ones((num_heads, q_len_chunk), dtype=torch.bool, device=value_states.device)
            # head_adaptive_capacity = torch.ones((bsz,num_heads),device=_device,dtype = sorted_attn_score_indices.dtype) * self.base_capacity
        # sorted_attn_score_indices = sorted_attn_score_indices.split(1,dim=1)

        heads_key_states = []
        heads_value_states = []
        assert bsz == 1
        # per head

        # reinit varlen metadata
        k_lens = []
        klen_sum = 0
        max_seqlen_k = 0
        self.cu_klen = 0


        for head_idx in range(num_heads):
            cache_index = torch.where(preserve_indices[head_idx])[0]
            off_set = torch.arange(self.chunk_size, device=cache_index.device).view((1, -1))
            cache_index = cache_index.view((-1, 1)) * self.chunk_size + off_set
            cache_index = cache_index[cache_index < q_len]
            cache_index = cache_index.flatten()

            l = cache_index.shape[-1] #+ self.window_size
            k_lens.append(l)
            max_seqlen_k = max(max_seqlen_k, l)
            klen_sum += l

            cache_index = cache_index.view(1, 1, -1, 1).expand(-1, -1, -1, head_dim)
            selected_k = origin_heads_key_states[head_idx].gather(dim=2, index=cache_index)
            selected_v = origin_heads_value_states[head_idx].gather(dim=2, index=cache_index)

            # NOTE: flatten view
            heads_key_states.append(selected_k.view(-1, head_dim))
            heads_value_states.append(selected_v.view(-1, head_dim))

        init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k)

        # NOTE: compose as flatten view
        heads_key_states = torch.cat(heads_key_states, dim=0)
        heads_value_states = torch.cat(heads_value_states, dim=0)
        return heads_key_states,heads_value_states


def init_snapkv(self):

    assert hasattr(self.config, 'window_size'), "window_size not set"
    assert hasattr(self.config, 'kernel_size'), "kernel_size not set"
    assert hasattr(self.config, "pooling"), "pooling not set"
    assert hasattr(self.config, "base_capacity"), "base_capacity not set"
    # init only once
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = SnapKVCluster(
            window_size = self.config.window_size, 
            max_capacity_prompt = self.config.base_capacity,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            layer_idx = self.layer_idx,
            num_hidden_layers = self.config.num_hidden_layers,
            pyram_mode = self.config.pyram_mode,
            pyram_beta = self.config.pyram_beta,
            gqa_support = self.config.gqa_support,
            num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads,
            gqa_func = self.config.gqa_func
            )
        if self.config.gqa_support:
            if self.config.model_type != "mistral":
                warnings.warn("GQA currently supports only for mistral-7B-v0.2 model")
        # if len(self.config.skip) > 0:
        #     warnings.warn("vanilla transformer should not enable skip",self.config.skip)
        print(f"Compress config(Snap): window_size={self.kv_cluster.window_size}, max_capacity_prompt={self.kv_cluster.max_capacity_prompt}, kernel_size={self.kv_cluster.kernel_size}, pooling={self.kv_cluster.pooling}, pyram_mode={self.kv_cluster.pyram_mode}, beta={self.kv_cluster.pyram_beta}",  flush=True)

def init_andpro_snapkv(self):
    assert hasattr(self.config,'window_size'),"window_size not set"
    assert hasattr(self.config,'kernel_size'),"kernel_size not set"
    assert hasattr(self.config,"pooling"),"pooling not set"
    assert hasattr(self.config, "base_capacity"), "base_capacity not set"
    assert hasattr(self.config,"floor_alpha"),"floor_alpha not set"
    assert hasattr(self.config,"chunk_size"),"chunk_size not set"
    assert hasattr(self.config,"mode"),"mode not set"
    assert self.config.floor_alpha is not None


    # init only once
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = AnDProAdaptiveSnapKVCluster(
            window_size = self.config.window_size,
            base_capacity=self.config.base_capacity,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            floor_alpha= self.config.floor_alpha,
            skip = self.config.skip,
            layer_idx = self.layer_idx,
            normalize = self.config.normalize,
            num_hidden_layers = self.config.num_hidden_layers,
            pyram_mode = self.config.pyram_mode,
            pyram_beta = self.config.pyram_beta,
            gqa_support = self.config.gqa_support,
            num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads,
            gqa_func = self.config.gqa_func,
            chunk_size = self.config.chunk_size,
            mode = self.config.mode,
            lambda_red = getattr(self.config, "lambda_red", 0.0),
            )
        if self.config.gqa_support:
            if self.config.model_type != "mistral":
                warnings.warn("GQA currently supports only for mistral-7B-v0.2 model")
        print(f"Compress config(Ada): window_size={self.kv_cluster.window_size}, base_capacity={self.kv_cluster.base_capacity}, kernel_size={self.kv_cluster.kernel_size}, pooling={self.kv_cluster.pooling}, floor_alpha={self.kv_cluster.floor_ratio}, pyram_mode={self.kv_cluster.pyram_mode}, beta={self.kv_cluster.pyram_beta}", flush=True)



class StreamingLLMKVCluster():
    def __init__(self, max_capacity_prompt = 256):
        self.max_capacity_prompt = max_capacity_prompt - 4
        self.sink_token = 4
        assert self.max_capacity_prompt - 4 > 0


    def update_kv(self, key_states, query_states, value_states, *args, **kwargs):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        print(f"StreamingLLM max_capacity_prompt {self.max_capacity_prompt + 4}")
        
        if q_len < self.max_capacity_prompt + 4:
            return key_states, value_states
        else:
            k_past_compress = key_states[:, :, :self.sink_token, :]
            v_past_compress = value_states[:, :, :self.sink_token, :]
            k_cur = key_states[:, :, -self.max_capacity_prompt:, :]
            v_cur = value_states[:, :, -self.max_capacity_prompt:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states


def init_slm(self,**kwargs):
    assert hasattr(self.config, 'window_size'), "window_size not set"
    # init only once
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = StreamingLLMKVCluster(
            max_capacity_prompt = self.config.base_capacity,
            )
        print(f"Compress config(SLM): max_cap={self.config.base_capacity}")





