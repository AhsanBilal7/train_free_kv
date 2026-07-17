import math
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache


# ------------------------------------------------------------------
# Flattened per-head cache, identical contract to
# causal_andpro_snapkv.monkeypatch.snapkv_utils.DynamicCacheSplitHeadFlatten.
# Duplicated here (rather than imported) so this package has no dependency
# on causal_andpro_snapkv and can be copied/moved independently.
# ------------------------------------------------------------------
class DynamicCacheSplitHeadFlatten(Cache):
    def __init__(self) -> None:
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0

    def __len__(self):
        return len(self.key_cache)

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (tuple(self.key_cache[layer_idx]), tuple(self.value_cache[layer_idx]))

    def __getitem__(self, layer_idx: int) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        if layer_idx < len(self):
            return (tuple(self.key_cache[layer_idx]), tuple(self.value_cache[layer_idx]))
        raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            assert self.key_cache[layer_idx].dim() == 2
            bs, head, seqlen, dim = key_states.shape
            assert bs == 1 and seqlen == 1
            head_lens = cache_kwargs["head_lens"]
            cu_klen = cache_kwargs["cu_klen"]
            from tiny_api_cuda import update_flatten_view
            new_key_cache = update_flatten_view(self.key_cache[layer_idx].view(-1, dim), key_states.view(-1, dim), head_lens, cu_klen)
            new_value_cache = update_flatten_view(self.value_cache[layer_idx].view(-1, dim), value_states.view(-1, dim), head_lens, cu_klen)
            self.key_cache[layer_idx] = new_key_cache
            self.value_cache[layer_idx] = new_value_cache
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        return 1

    def get_max_length(self) -> Optional[int]:
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values=None) -> "DynamicCacheSplitHeadFlatten":
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class TestTimeLookaheadKVCluster:
    """Implements "Training-Free Test-Time Lookahead Causal KV Eviction" (proposed_updated_idea.pdf).

    Unlike AnDProAdaptiveSnapKVCluster.update_kv (causal_andpro_snapkv), compression here is NOT
    triggered automatically inside the attention forward during prefill. The attention hijack only
    feeds this object raw observations (window queries, lookahead-trajectory queries, the full
    uncompressed K/V); the actual scoring + eviction happens once, in `compress()`, called explicitly
    by causal_lookahead_snapkv.monkeypatch.lookahead_runner after all M lookahead particles have run
    (Algorithm 1, lines 1-34). This split is required because the lookahead continuations themselves
    must be generated against the *full* (uncompressed) cache before eviction can happen.
    """

    def __init__(self, window_size, base_capacity, chunk_size, skip, layer_idx, num_hidden_layers,
                 num_particles, lookahead_len, lambda_h, particle_eta,
                 robust_alpha, robust_topk_beta, lambda_lookahead, lambda_k, lambda_v, budget_tau,
                 lookahead_temperature, causal_eps=1e-6, candidate_chunk=8192):
        self.window_size = window_size
        self.base_capacity = base_capacity
        self.chunk_size = chunk_size
        self.skip = skip
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers

        # Eq. 4-8: lookahead particle generation + confidence weighting.
        self.num_particles = num_particles
        self.lookahead_len = lookahead_len
        self.lambda_h = lambda_h
        self.particle_eta = particle_eta
        self.lookahead_temperature = lookahead_temperature

        # Eq. 15-18: robust aggregation + particle-weighted lookahead score.
        self.robust_alpha = robust_alpha
        self.robust_topk_beta = robust_topk_beta
        self.lambda_lookahead = lambda_lookahead

        # Eq. 19-21: adaptive head-wise budget + diversity-aware greedy selection.
        self.budget_tau = budget_tau
        self.lambda_k = lambda_k
        self.lambda_v = lambda_v

        self.causal_eps = causal_eps
        self.candidate_chunk = candidate_chunk

        # Filled in by the attention hijack during the "collect" phase.
        self.o_proj_weight = None
        self.head_dim = None
        self.num_key_value_groups = 1       # num_heads // num_key_value_heads; set by init_lookahead_snapkv
        self.window_queries = None          # [1, H, window_size, D]  (H = num_heads, queries are never GQA-repeated)
        self.full_key_states = None         # [1, KV_H, n, D]  COMPACT -- not GQA-repeated, to avoid
        self.full_value_states = None       # [1, KV_H, n, D]  persisting a much larger tensor across
                                             # the whole lookahead rollout. compress() repeats_kv
                                             # transiently, one layer at a time, right before scoring.

        # Filled in during "lookahead" phase, one list of [1,H,1,D] tensors per particle.
        self._lookahead_buffers = [[] for _ in range(num_particles)]
        self.particle_weights = None        # tensor [M], set after particle confidence scoring

        # Decode-time (varlen flash-attn) metadata, same convention as causal_andpro_snapkv.
        self.head_lens = None
        self.max_seqlen_k = 0
        self.klen_sum = 0
        self.cu_klen = 0
        self.cu_offset = None
        self.cu_head_offset = None
        self.cu_qlen = None
        self.qlen_sum = None
        self.layer_qlens = None

    # ------------------------------------------------------------------
    # Observation hooks called from the attention forward hijack.
    # ------------------------------------------------------------------
    def capture_window_queries(self, query_states):
        """Eq. 2/5: Q_W, the last `window_size` prefill queries (post-RoPE)."""
        w = min(self.window_size, query_states.shape[2])
        self.window_queries = query_states[:, :, -w:, :].detach()

    def store_full_kv(self, key_states, value_states):
        """Full, uncompressed prefill K/V (Eq. 1), kept alive across the whole lookahead rollout."""
        self.full_key_states = key_states.detach()
        self.full_value_states = value_states.detach()

    def capture_lookahead_query(self, particle_idx, query_states):
        """Eq. 5: one query state from the m-th lookahead trajectory."""
        self._lookahead_buffers[particle_idx].append(query_states.detach())

    def set_particle_weights(self, weights):
        """Eq. 8: normalized particle confidence weights omega_m."""
        self.particle_weights = weights

    def release_collected_state(self):
        """Drop the full, uncompressed prefill K/V and window queries once compress() has
        consumed them. Without this, every layer keeps its (potentially 64K-token) full K/V
        alive for the rest of the document's decode loop -- and into the next document's collect
        phase -- which is the single largest source of peak memory in this pipeline."""
        self.full_key_states = None
        self.full_value_states = None
        self.window_queries = None

    def reset_lookahead_buffers(self):
        self._lookahead_buffers = [[] for _ in range(self.num_particles)]
        self.particle_weights = None

    # ------------------------------------------------------------------
    # Eq. 9-17: output-projection-aware causal score, robustly aggregated over a query batch.
    # ------------------------------------------------------------------
    def _robust_topbeta_mean(self, scores):
        """scores: [Q, chunk] -> [chunk]. Eq. 16-17: alpha * TopAvg_beta + (1-alpha) * Mean, over Q."""
        q = scores.shape[0]
        k = max(1, math.ceil(self.robust_topk_beta * q))
        topk_vals, _ = scores.topk(k, dim=0)
        topavg = topk_vals.mean(dim=0)
        mean_ = scores.mean(dim=0)
        return self.robust_alpha * topavg + (1.0 - self.robust_alpha) * mean_

    def _project_and_score(self, key_states, value_states, query_batches, causal_flags):
        """Eq. 9-14 scoring, decomposed via the W_O-projection Gram-matrix identity so the
        [Q, n, hidden] perturbation tensor is never materialized:

            delta_y_{q,i} = c_{q,i} * (v_i - y_q),         c_{q,i} = a_{q,i} / (1 - a_{q,i} + eps)
            delta_o_{q,i} = W_O @ delta_y_{q,i} = c_{q,i} * (W_O v_i - W_O y_q)
            s_{q,i}       = |c_{q,i}| * || W_O v_i - W_O y_q ||_2
                          = |c_{q,i}| * sqrt(||Wv_i||^2 + ||Wy_q||^2 - 2 Wv_i . Wy_q)

        `Wv_i = W_O v_i` depends only on the (shared) value states, so it is computed once per
        head and reused across the window batch and every lookahead particle's batch.

        key_states, value_states: [1, H, n, D] (already GQA-repeated to query-head count).
        query_batches: list of [1, H, Qi, D] tensors.
        causal_flags: list of bool, True only for the prompt's own observation-window batch
            (its queries must not attend to *later* window positions; every other candidate is
            already in the causal past). Lookahead queries postdate the whole prompt, so no mask.
        Returns: list of [H, n] float32 robust-aggregated scores, S_i,W and S_i,m1..S_i,mM.
        """
        H = key_states.shape[1]
        n = key_states.shape[2]
        D = self.head_dim
        device = key_states.device

        value_f = value_states.squeeze(0).float()  # [H, n, D]
        key_f = key_states.squeeze(0).float()      # [H, n, D]

        out_scores = [torch.zeros(H, n, device=device, dtype=torch.float32) for _ in query_batches]

        for h in range(H):
            Wo_h = self.o_proj_weight[:, h * D:(h + 1) * D].float()  # [hidden, D]
            v_h = value_f[h]   # [n, D]
            k_h = key_f[h]     # [n, D]
            Wv_h = v_h @ Wo_h.T               # [n, hidden]
            normsq_v_h = (Wv_h * Wv_h).sum(-1)  # [n]

            for b, (q_batch, causal) in enumerate(zip(query_batches, causal_flags)):
                q_h = q_batch[0, h].float()  # [Q, D]
                Q = q_h.shape[0]
                attn_logits = (q_h @ k_h.T) / math.sqrt(D)  # [Q, n]
                if causal:
                    w = min(Q, n)
                    mask = torch.full((w, w), float('-inf'), device=device)
                    mc = torch.arange(w, device=device)
                    mask.masked_fill_(mc <= mc.view(-1, 1), 0.0)
                    attn_logits[:, -w:] = attn_logits[:, -w:] + mask
                attn_w = F.softmax(attn_logits, dim=-1)  # [Q, n] : a_{q,i}
                y_h = attn_w @ v_h                          # [Q, D] : y_q
                Wy_h = y_h @ Wo_h.T                          # [Q, hidden]
                normsq_y_h = (Wy_h * Wy_h).sum(-1)           # [Q]
                c = attn_w / (1.0 - attn_w + self.causal_eps)  # [Q, n]

                for start in range(0, n, self.candidate_chunk):
                    end = min(start + self.candidate_chunk, n)
                    dot = Wy_h @ Wv_h[start:end].T  # [Q, chunk]
                    dist_sq = normsq_y_h.unsqueeze(1) + normsq_v_h[start:end].unsqueeze(0) - 2 * dot
                    dist = dist_sq.clamp_min(0).sqrt()
                    out_scores[b][h, start:end] = self._robust_topbeta_mean(c[:, start:end].abs() * dist)
        return out_scores

    # ------------------------------------------------------------------
    # Eq. 19-22: adaptive head-wise budget allocation + diversity-aware greedy selection.
    # ------------------------------------------------------------------
    def _select_chunks(self, token_score, n, H, key_states, value_states):
        device = token_score.device
        pad_len = (self.chunk_size - (n % self.chunk_size)) % self.chunk_size
        score_padded = F.pad(token_score, (0, pad_len), value=float('-inf'))
        n_chunk = (n + pad_len) // self.chunk_size
        chunk_score = score_padded.view(H, n_chunk, self.chunk_size).amax(dim=-1)

        drop = torch.ones((H, n_chunk), dtype=torch.bool, device=device)
        drop[:, 0] = False
        drop[:, -self.window_size // self.chunk_size:] = False
        num_preserved_per_head = self.window_size // self.chunk_size + 1

        zeros = torch.zeros_like(chunk_score)
        masked_for_mean = torch.where(drop, chunk_score, zeros)
        mean_per_head = masked_for_mean.sum(-1) / drop.sum(-1).clamp(min=1)
        normalized = chunk_score / (mean_per_head.unsqueeze(-1) + self.causal_eps)
        normalized = normalized.masked_fill(~drop, float('-inf'))

        # Eq. 19: per-head perturbation mass, from the normalized droppable chunk scores.
        E = torch.where(drop, normalized.clamp_min(0), zeros).sum(-1)  # [H]
        total_chunks_per_head = self.base_capacity // self.chunk_size
        total_budget = H * total_chunks_per_head - H * num_preserved_per_head
        weight = (E + self.causal_eps).pow(self.budget_tau)
        raw_alloc = total_budget * weight / weight.sum()
        head_budget = raw_alloc.floor().long()
        # Eq. 20's floor() can undershoot the global budget by a few chunks; hand the remainder
        # to the heads with the largest fractional part so the total stays exactly on-budget.
        deficit = int(total_budget - head_budget.sum().item())
        if deficit > 0:
            order = torch.argsort(raw_alloc - head_budget.float(), descending=True)
            head_budget[order[:deficit]] += 1

        if self.lambda_k > 0 or self.lambda_v > 0:
            preserve = self._greedy_diverse_select(normalized, drop, head_budget, n, n_chunk, key_states, value_states)
        else:
            preserve = ~drop.clone()
            for h in range(H):
                b = int(head_budget[h].item())
                if b <= 0:
                    continue
                idx = torch.topk(normalized[h], k=b, dim=-1).indices
                preserve[h, idx] = True
        return preserve

    def _greedy_diverse_select(self, normalized, drop, head_budget, n, n_chunk, key_states, value_states):
        """Eq. 21: i* = argmax_i [ S_i - lambda_k*max_j cos(k_i,k_j) - lambda_v*max_j cos(v_i,v_j) ],
        applied chunk-wise (chunks represented by their mean-pooled, L2-normalized key/value), with
        a per-head budget (Eq. 20) instead of one global flat top-K."""
        H = normalized.shape[0]
        device = normalized.device
        D = self.head_dim
        pad_len = n_chunk * self.chunk_size - n

        k = key_states.squeeze(0).float()
        v = value_states.squeeze(0).float()
        if pad_len > 0:
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
        k_repr = k.view(H, n_chunk, self.chunk_size, D).mean(dim=2)
        v_repr = v.view(H, n_chunk, self.chunk_size, D).mean(dim=2)
        k_unit = k_repr / (k_repr.norm(dim=-1, keepdim=True) + self.causal_eps)
        v_unit = v_repr / (v_repr.norm(dim=-1, keepdim=True) + self.causal_eps)

        drop = drop.clone()
        preserved = ~drop
        max_pen = torch.full((H, n_chunk), float('-inf'), device=device)
        for h in range(H):
            sel = torch.where(preserved[h])[0]
            if sel.numel() == 0:
                continue
            pen = self.lambda_k * (k_unit[h] @ k_unit[h, sel].T) + self.lambda_v * (v_unit[h] @ v_unit[h, sel].T)
            max_pen[h] = pen.amax(dim=-1)

        for h in range(H):
            b = int(head_budget[h].item())
            if b <= 0:
                continue
            eligible = drop[h].clone()
            for _ in range(b):
                cand = normalized[h] - max_pen[h]
                cand = cand.masked_fill(~eligible, float('-inf'))
                idx = int(cand.argmax().item())
                drop[h, idx] = False
                eligible[idx] = False
                pen_new = self.lambda_k * (k_unit[h] @ k_unit[h, idx]) + self.lambda_v * (v_unit[h] @ v_unit[h, idx])
                max_pen[h] = torch.maximum(max_pen[h], pen_new)
        return ~drop

    # ------------------------------------------------------------------
    # Top-level entry point, called once per layer by lookahead_runner after all M particles
    # have been rolled out (Algorithm 1, lines 9-34).
    # ------------------------------------------------------------------
    def compress(self):
        # GQA-repeat here, transiently, instead of persisting the repeated (num_heads-count, up to
        # 4x larger for Mistral) tensor on self.full_key_states/full_value_states for the whole
        # lookahead rollout -- this local is freed as soon as compress() returns.
        key_states = repeat_kv(self.full_key_states, self.num_key_value_groups)
        value_states = repeat_kv(self.full_value_states, self.num_key_value_groups)
        H = key_states.shape[1]
        n = key_states.shape[2]
        D = self.head_dim

        if self.layer_idx < self.skip or n <= self.base_capacity:
            heads_key_states = key_states.reshape(-1, D)
            heads_value_states = value_states.reshape(-1, D)
            return self._finalize_metadata(heads_key_states, heads_value_states, [n] * H, key_states.device)

        batches = [self.window_queries]
        causal_flags = [True]
        for m in range(self.num_particles):
            buf = self._lookahead_buffers[m]
            q_m = torch.cat(buf, dim=2) if len(buf) > 0 else self.window_queries[:, :, :0, :]
            batches.append(q_m)
            causal_flags.append(False)

        scores = self._project_and_score(key_states, value_states, batches, causal_flags)
        S_total = scores[0]  # Eq. 18 first term: S_i,W
        if self.lambda_lookahead > 0 and self.particle_weights is not None:
            for m in range(self.num_particles):
                S_total = S_total + self.lambda_lookahead * self.particle_weights[m] * scores[1 + m]

        preserve_indices = self._select_chunks(S_total, n, H, key_states, value_states)

        origin_heads_key_states = torch.split(key_states, 1, dim=1)
        origin_heads_value_states = torch.split(value_states, 1, dim=1)

        heads_key_states, heads_value_states, k_lens = [], [], []
        for head_idx in range(H):
            cache_index = torch.where(preserve_indices[head_idx])[0]
            off_set = torch.arange(self.chunk_size, device=cache_index.device).view((1, -1))
            cache_index = cache_index.view((-1, 1)) * self.chunk_size + off_set
            cache_index = cache_index[cache_index < n]
            cache_index = cache_index.flatten()
            k_lens.append(cache_index.shape[-1])

            cache_index_exp = cache_index.view(1, 1, -1, 1).expand(-1, -1, -1, D)
            selected_k = origin_heads_key_states[head_idx].gather(dim=2, index=cache_index_exp)
            selected_v = origin_heads_value_states[head_idx].gather(dim=2, index=cache_index_exp)
            heads_key_states.append(selected_k.reshape(-1, D))
            heads_value_states.append(selected_v.reshape(-1, D))

        heads_key_states = torch.cat(heads_key_states, dim=0)
        heads_value_states = torch.cat(heads_value_states, dim=0)
        return self._finalize_metadata(heads_key_states, heads_value_states, k_lens, key_states.device)

    def _finalize_metadata(self, heads_key_states, heads_value_states, k_lens, device):
        H = len(k_lens)
        self.head_lens = torch.tensor(k_lens, dtype=torch.int32, device=device)
        self.klen_sum = int(sum(k_lens))
        self.max_seqlen_k = int(max(k_lens))
        self.cu_headlens = torch.cumsum(self.head_lens, dim=0, dtype=torch.int32)
        self.cu_klen = self.cu_headlens - self.head_lens
        self.cu_klen = torch.cat([self.cu_klen, torch.tensor([self.klen_sum], dtype=torch.int32, device=device)], dim=0)
        self.layer_qlens = torch.ones(H, dtype=torch.int32, device=device)
        self.qlen_sum = H
        self.cu_qlen = torch.cumsum(self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
        self.cu_qlen = torch.cat([self.cu_qlen, torch.tensor([H], dtype=torch.int32, device=device)], dim=0)
        self.cu_offset = torch.arange(0, H + 1, dtype=torch.int32, device=device)
        self.cu_head_offset = torch.arange(1, H + 1, dtype=torch.int32, device=device)
        cache_kwargs = {"head_lens": self.head_lens, "cu_klen": self.cu_klen}
        return heads_key_states, heads_value_states, cache_kwargs


def init_lookahead_snapkv(self):
    assert hasattr(self.config, "window_size"), "window_size not set"
    assert hasattr(self.config, "base_capacity"), "base_capacity not set"
    assert hasattr(self.config, "chunk_size"), "chunk_size not set"
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = TestTimeLookaheadKVCluster(
            window_size=self.config.window_size,
            base_capacity=self.config.base_capacity,
            chunk_size=self.config.chunk_size,
            skip=self.config.skip,
            layer_idx=self.layer_idx,
            num_hidden_layers=self.config.num_hidden_layers,
            num_particles=self.config.num_particles,
            lookahead_len=self.config.lookahead_len,
            lambda_h=self.config.lambda_h,
            particle_eta=self.config.particle_eta,
            robust_alpha=self.config.robust_alpha,
            robust_topk_beta=self.config.robust_topk_beta,
            lambda_lookahead=self.config.lambda_lookahead,
            lambda_k=self.config.lambda_k,
            lambda_v=self.config.lambda_v,
            budget_tau=self.config.budget_tau,
            lookahead_temperature=self.config.lookahead_temperature,
        )
        self.kv_cluster.o_proj_weight = self.o_proj.weight
        self.kv_cluster.head_dim = self.head_dim
        self.kv_cluster.num_key_value_groups = self.num_key_value_groups
        print(f"Compress config(Lookahead): window_size={self.kv_cluster.window_size}, "
              f"base_capacity={self.kv_cluster.base_capacity}, chunk_size={self.kv_cluster.chunk_size}, "
              f"M={self.kv_cluster.num_particles}, T={self.kv_cluster.lookahead_len}, "
              f"lambda_lookahead={self.kv_cluster.lambda_lookahead}, lambda_k={self.kv_cluster.lambda_k}, "
              f"lambda_v={self.kv_cluster.lambda_v}, budget_tau={self.kv_cluster.budget_tau}", flush=True)
