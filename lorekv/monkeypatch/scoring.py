"""LORE-KV scoring, reliability weights and token selection (paper Secs. 3.2-3.5).

The notation follows the paper. For layer l and head h, K and V are the prefill
keys and values. Q_W are the queries of the last w prompt tokens. Q_look^(m) are
the queries of lookahead trajectory m = 1, ..., M.

Contents:
    step_stats: Centered score c_u (Eq. 2), entropy H_u and varentropy V_u (Eq. 9)
        of one sampled token.
    reliability_weights: Reliability R_m (Eq. 3) and weights omega (Eq. 4).
    divergence_from_query_stacks, standardize_disagreement: Query-space
        disagreement D_m (Eq. 10) and its standardized value D_hat (App. C.3).
    TestTimeLookaheadKVCluster: Per-layer state and eviction. It computes the
        projected deletion scores (Eqs. 5-6), the token score S_i (Eq. 7), the head
        budgets and the selection (Eq. 8, Eq. 11, App. D).

Selection uses chunks of ``chunk_size`` adjacent tokens. The score of a chunk is
the maximum score of its tokens (paper Table 4: chunk size 4).
"""
import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache


class DynamicCacheSplitHeadFlatten(Cache):
    """KV cache in which each head can keep a different number of tokens.

    Each layer stores its heads one after the other in one tensor of shape
    [sum_h len_h, head_dim]. ``head_lens`` and the offsets ``cu_klen`` give the
    position of each head. ``flash_attn_varlen_func`` reads this layout during
    decoding. The CUDA kernel ``update_flatten_view`` (``csrc/``) appends one token to
    each head. Adapted from AdaKV (https://github.com/FFY0/AdaKV, MIT).
    """

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


# ---------------------------------------------------------------------------
# Trajectory reliability (Sec. 3.3, App. C.3).
#
# For a token y_u sampled from q(.|h_u), the centered score (Eq. 2) is
#
#     c_u = log q(y_u | h_u) + H(q(. | h_u)),        E[c_u | h_u] = 0.
#
# c_u measures how atypical the sampled token is. H measures how uncertain the
# context is. The trajectory reliability (Eq. 3) is
#
#     R_m = beta * c_bar_m - gamma * H_bar_m - lambda_D * D_hat_m,   omega = softmax(R).
#
# mean_u log q = c_bar - H_bar. Thus (0, 0, 0) gives uniform weights and (1, 1, 0)
# gives log-probability weights (App. C.3). LORE-KV /E uses (1, 1.3, 0) and /D uses
# (1, 1, 0.3) (Table 4). The softmax has no temperature, because the three
# coefficients set the scale.
#
# E[c_u | h_u] = 0 only if c_u is centered on the sampling distribution. Apply each
# temperature and truncation to the logits before the log-softmax. The paper
# samples at temperature 1 without truncation (App. E.1).
# ---------------------------------------------------------------------------
def mask_top_k(z, top_k):
    """Set the logits outside the top-k tokens to -inf."""
    if not top_k or top_k <= 0 or top_k >= z.shape[-1]:
        return z
    kth = z.topk(int(top_k), dim=-1).values[..., -1:]
    return z.masked_fill(z < kth, float("-inf"))


def mask_top_p(z, top_p):
    """Set the logits outside the top-p nucleus to -inf."""
    if top_p is None or top_p >= 1.0:
        return z
    srt, idx = z.sort(dim=-1, descending=True)
    cum = srt.softmax(dim=-1).cumsum(dim=-1)
    drop = cum - srt.softmax(dim=-1) >= float(top_p)   # keep the token that crosses top_p
    drop[..., 0] = False
    return z.masked_fill(drop.scatter(-1, idx, drop), float("-inf"))


def step_stats(logits, y, temperature=1.0, top_k=None, top_p=None):
    """Compute the reliability statistics of one sampled token (Eqs. 2 and 9).

    Use the same temperature and truncation as the sampler. Otherwise the centering
    distribution is not the sampling distribution, and E[c | h] is not zero.

    Args:
        logits: Tensor [M, V]. The raw next-token logits that the sampler used.
        y: Tensor [M]. The sampled token IDs.
        temperature: Sampling temperature.
        top_k: Optional top-k truncation.
        top_p: Optional top-p truncation.

    Returns:
        A tuple ``(logp, H, V, c)`` of tensors [M]: log q(y | h), the entropy of
        q(. | h) in nats, the varentropy Var_{y~q}[log q(y)], and the centered score
        c = logp + H.
    """
    z = logits.float() / max(float(temperature), 1e-5)
    if top_k is not None:
        z = mask_top_k(z, top_k)
    if top_p is not None:
        z = mask_top_p(z, top_p)

    logq = torch.log_softmax(z, dim=-1)          # centering law == sampling law
    q = logq.exp()

    # Mask the products: outside a truncated support q = 0 and log q = -inf, and
    # 0 * -inf is NaN.
    zeros = torch.zeros_like(q)
    qlogq = torch.where(q > 0, q * logq, zeros)
    qlogq2 = torch.where(q > 0, q * logq.pow(2), zeros)

    H = -qlogq.sum(-1)                            # entropy
    V = (qlogq2.sum(-1) - H.pow(2)).clamp_min(0)  # varentropy = Var[log q]
    lp = logq.gather(-1, y.reshape(-1, 1).to(logq.device)).squeeze(-1)
    return lp, H, V, lp + H


def reliability_weights(c, H, V, D_hat, beta, gamma, lambda_d=0.0, normalize=False):
    """Compute the trajectory weights omega (Eqs. 3-4).

    Args:
        c: Tensor [M, T]. Centered scores.
        H: Tensor [M, T]. Entropies.
        V: Tensor [M, T]. Varentropies.
        D_hat: Tensor [M] with the standardized disagreement, or None.
        beta: Coefficient of the mean centered score.
        gamma: Coefficient of the mean entropy.
        lambda_d: Coefficient of the disagreement.
        normalize: If True, use c_bar / sqrt(V_bar / T) in place of c_bar (App. C.3,
            ablation only).

    Returns:
        A tuple ``(omega, parts)``. ``omega`` is a tensor [M] that sums to one.
        ``parts`` contains the per-trajectory values c_bar, H_bar, V_bar and R.
    """
    T = max(int(c.shape[-1]), 1)
    c_bar, H_bar, V_bar = c.mean(-1), H.mean(-1), V.mean(-1)
    x = c_bar / (V_bar / T).clamp_min(1e-8).sqrt() if normalize else c_bar
    if D_hat is None:
        D_hat = torch.zeros_like(c_bar)
    R = beta * x - gamma * H_bar - lambda_d * D_hat
    return torch.softmax(R, dim=0), {
        "c_bar": c_bar, "H_bar": H_bar, "V_bar": V_bar, "typicality": x, "R": R,
    }


# ---------------------------------------------------------------------------
# Query-space disagreement (Eq. 10). D_m is the mean cosine distance between the
# lookahead queries of trajectory m and those of the other trajectories:
#
#   D_m = 1/(M-1) sum_{r != m} 1/(T L H_q) sum_{u,l,g} [1 - cos(q^(m)_{u,l,g}, q^(r)_{u,l,g})].
#
# The two queries of a pair at step u have the same position n + u and the same
# rotary phase. After standardization D_hat sums to zero over the trajectories.
# It thus changes the weights inside the sample and does not move the target of
# the estimator (App. C.3).
# ---------------------------------------------------------------------------
@torch.no_grad()
def divergence_from_query_stacks(stacks, num_particles, eps=1e-6, device=None):
    """Compute the query-space disagreement D_m (Eq. 10).

    Args:
        stacks: Iterable of tensors [M, Hq, T, D], one for each layer. Entry m holds
            the post-RoPE lookahead queries of trajectory m.
        num_particles: Number of trajectories M.
        eps: Value that prevents division by zero.
        device: Device of the result. If None, the device of the first stack.

    Returns:
        A float32 tensor [M]. The tensor is zero if M <= 2 or if no layer has
        queries. For M = 2 the two values are equal and their standard deviation is
        zero, so lambda_D has no effect.
    """
    M = int(num_particles)
    if M <= 2:
        return torch.zeros(M, dtype=torch.float32, device=device)

    cos_sum = None
    n_groups = 0
    for q in stacks:
        if q is None:
            continue
        q = q.float()
        if device is None:
            device = q.device
        elif q.device != device:
            q = q.to(device)
        if cos_sum is None:
            cos_sum = torch.zeros(M, M, dtype=torch.float32, device=device)
        q = q / (q.norm(dim=-1, keepdim=True) + eps)
        # Mean cosine over heads and steps for each ordered pair of trajectories.
        cos_sum += torch.einsum("mhtd,rhtd->mr", q, q) / (q.shape[1] * q.shape[2])
        n_groups += 1

    if n_groups == 0:
        return torch.zeros(M, dtype=torch.float32, device=device)
    dist = 1.0 - cos_sum / n_groups                                 # [M, M]
    dist.fill_diagonal_(0.0)                                        # exclude r == m
    return dist.sum(dim=1) / (M - 1)


def _cluster_query_stacks(layers, num_particles, layer_stride=1):
    """Yield the lookahead queries of each compressed layer as a tensor [M, Hq, T, D]."""
    for layer in layers[::max(1, int(layer_stride))]:
        # The linear-attention layers of a hybrid backbone have no `self_attn`.
        _attn = getattr(layer, "self_attn", None)
        kvc = getattr(_attn, "kv_cluster", None) if _attn is not None else None
        if kvc is None:
            continue
        bufs = [kvc._lookahead_buffers[m] for m in range(num_particles)]
        if any(len(b) == 0 for b in bufs):
            continue
        # Compare only the steps that all trajectories produced.
        T = min(len(b) for b in bufs)
        yield torch.stack([torch.cat(b[:T], dim=2)[0] for b in bufs], dim=0)


@torch.no_grad()
def particle_trajectory_divergence(layers, num_particles, layer_stride=1, eps=1e-6, device=None):
    """Compute D_m (Eq. 10) from the lookahead buffers of the decoder layers.

    Call this function before ``reset_lookahead_buffers()``, which clears the buffers.

    Args:
        layers: The decoder layers.
        num_particles: Number of trajectories M.
        layer_stride: Use every ``layer_stride``-th layer.
        eps: Value that prevents division by zero.
        device: Device of the result.

    Returns:
        A float32 tensor [M].
    """
    return divergence_from_query_stacks(
        _cluster_query_stacks(layers, int(num_particles), layer_stride),
        int(num_particles), eps=eps, device=device)


def standardize_disagreement(x, eps=1e-6):
    """Return D_hat = (D - mean(D)) / sd(D) across the M trajectories (App. C.3).

    The standardization makes lambda_D independent of the prompt scale. The result
    is zero if M <= 2 or if all trajectories have the same disagreement.
    """
    if x.numel() <= 2:
        return torch.zeros_like(x)
    centered = x - x.mean()
    sd = x.std(unbiased=False)
    if float(sd) < eps:
        return torch.zeros_like(x)
    return centered / sd


# Alias used by the analysis figures.
standardize_particle_scores = standardize_disagreement


class TestTimeLookaheadKVCluster:
    """Per-layer state and eviction of LORE-KV (Algorithm 1, lines 7-24).

    Each compressed attention layer has one instance (``self.kv_cluster``). The
    baselines evict during the prefill forward. LORE-KV must first generate the
    lookahead trajectories with the full cache. The attention forward thus only
    records Q_W, the queries of each trajectory and the full K and V.
    :func:`lorekv.monkeypatch.lookahead_runner.generate_with_lookahead` then calls
    :meth:`compress` one time for each layer.
    """

    def __init__(self, window_size, base_capacity, chunk_size, skip, layer_idx, num_hidden_layers,
                 num_particles, lookahead_len, particle_beta, particle_gamma,
                 robust_alpha, robust_topk_beta, lambda_lookahead, lambda_k, lambda_v, budget_tau,
                 lookahead_temperature, lambda_d=0.0, divergence_layer_stride=1,
                 normalize_typicality=False, score_objective="projected",
                 causal_eps=1e-6, candidate_chunk=8192):
        self.window_size = window_size
        self.base_capacity = base_capacity
        self.chunk_size = chunk_size
        self.skip = skip
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers

        # Lookahead (Eq. 1) and reliability R_m = beta*c_bar - gamma*H_bar - lambda_D*D_hat (Eq. 3).
        self.num_particles = num_particles
        self.lookahead_len = lookahead_len
        self.particle_beta = particle_beta   # beta:  typicality (centered log-prob) coefficient
        self.particle_gamma = particle_gamma # gamma: entropy coefficient
        self.lambda_d = lambda_d             # lambda_D: trajectory-disagreement coefficient
        self.normalize_typicality = bool(normalize_typicality)  # App. C.3 ablation
        self.divergence_layer_stride = divergence_layer_stride
        self.lookahead_temperature = lookahead_temperature

        # Aggregation psi (App. D-A: alpha, kappa) and lookahead weight lambda_L (Eq. 7).
        self.robust_alpha = robust_alpha
        self.robust_topk_beta = robust_topk_beta
        self.lambda_lookahead = lambda_lookahead

        # Head budgets (Eq. 11: tau) and diversity-aware selection (App. D-C: lambda_k, lambda_v).
        self.budget_tau = budget_tau
        self.lambda_k = lambda_k
        self.lambda_v = lambda_v

        # Per-query deletion score (App. D, ablation iii):
        #   'projected'   |c_{q,i}| * || W_O v_i - W_O o_q ||   (Eqs. 5-6, the core method)
        #   'unprojected' |c_{q,i}| * ||      v_i -      o_q ||   (W_O dropped)
        #   'attention'   a_{q,i}                                (accumulated attention mass)
        self.score_objective = str(score_objective).lower()
        if self.score_objective not in ("projected", "unprojected", "attention"):
            raise ValueError(
                f"score_objective must be projected|unprojected|attention, got {score_objective!r}")

        self.causal_eps = causal_eps
        self.candidate_chunk = candidate_chunk

        # Set by the attention forward in the "collect" phase.
        self.o_proj_weight = None
        self.head_dim = None
        self.num_key_value_groups = 1       # num_heads // num_key_value_heads; set by init_lookahead_snapkv
        self.window_queries = None          # [1, H, window_size, D]  (H = num_heads, queries are never GQA-repeated)
        self.full_key_states = None         # [1, KV_H, n, D]  COMPACT -- not GQA-repeated, to avoid
        self.full_value_states = None       # [1, KV_H, n, D]  persisting a much larger tensor across
                                             # the whole lookahead rollout. compress() repeats_kv
                                             # transiently, one layer at a time, right before scoring.

        # Set in the "lookahead" phase: one list of [1, H, 1, D] tensors per trajectory.
        self._lookahead_buffers = [[] for _ in range(num_particles)]
        self.particle_weights = None        # tensor [M], set after particle confidence scoring

        # Metadata for the varlen flash-attention decode step (as in the baselines).
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
    # Observation hooks. The attention forward calls them.
    # ------------------------------------------------------------------
    def capture_window_queries(self, query_states):
        """Record Q_W, the post-RoPE queries of the last ``window_size`` prompt tokens."""
        w = min(self.window_size, query_states.shape[2])
        # Clone the slice. A view keeps the full [1, H, n, D] query tensor of each layer
        # in memory until compress() (32 GiB for Llama-3.1-8B at 128K tokens).
        self.window_queries = query_states[:, :, -w:, :].detach().clone()

    def store_full_kv(self, key_states, value_states):
        """Record the full prefill K and V (KV-head count) until :meth:`compress`."""
        self.full_key_states = key_states.detach()
        self.full_value_states = value_states.detach()

    def capture_lookahead_query(self, particle_idx, query_states):
        """Append the query of one step of lookahead trajectory ``particle_idx``."""
        self._lookahead_buffers[particle_idx].append(query_states.detach())

    def set_particle_weights(self, weights):
        """Set the trajectory weights omega (Eq. 4)."""
        self.particle_weights = weights

    def release_collected_state(self):
        """Release the full K, V and Q_W after :meth:`compress`.

        Without this step each layer keeps its full prefill cache during decoding. This
        is the largest contribution to the peak memory.
        """
        self.full_key_states = None
        self.full_value_states = None
        self.window_queries = None

    def reset_lookahead_buffers(self):
        self._lookahead_buffers = [[] for _ in range(self.num_particles)]
        self.particle_weights = None

    # ------------------------------------------------------------------
    # Deletion scores (Eqs. 5-6) aggregated over a query set (psi in Eq. 7).
    # ------------------------------------------------------------------
    def _robust_topbeta_mean(self, scores):
        """Aggregate scores over queries: psi(E) = alpha * TopAvg_kappa(E) + (1 - alpha) * Mean(E).

        The core method uses alpha = 0, which gives the mean (App. D-A). ``robust_topk_beta``
        is kappa. The rule follows DefensiveKV (Feng et al., 2025a).

        Args:
            scores: Tensor [Q, n_chunk].

        Returns:
            Tensor [n_chunk]. Zero if Q = 0.
        """
        q = scores.shape[0]
        if q == 0:
            # A trajectory with no captured queries (compress() passes an empty batch
            # for it) carries no information and contributes zero.
            return scores.new_zeros(scores.shape[1:])
        k = max(1, math.ceil(self.robust_topk_beta * q))
        topk_vals, _ = scores.topk(k, dim=0)
        topavg = topk_vals.mean(dim=0)
        mean_ = scores.mean(dim=0)
        return self.robust_alpha * topavg + (1.0 - self.robust_alpha) * mean_

    def _project_and_score(self, key_states, value_states, query_batches, causal_flags):
        """Compute the aggregated leave-one-out deletion scores for each query set (Eqs. 5-7).

        a_{q,i} is the full-cache attention weight and o_q is the full-cache output. If
        token i is deleted and the weights are normalized again, the output changes by

            delta_o_{q,i} = c_{q,i} * (v_i - o_q),    c_{q,i} = a_{q,i} / (1 - a_{q,i} + eps).

        The score measures this change after the output projection W_O of the head
        (Eq. 6). The function computes the norm with the Gram identity

            ||W_O v_i - W_O o_q||^2 = ||W_O v_i||^2 + ||W_O o_q||^2 - 2 <W_O v_i, W_O o_q>,

        so it does not make the [Q, n, hidden] tensor. Attention is computed only for the
        w + M*T scoring queries (App. C.1). Prefill and decoding keep their fused kernels.

        ``score_objective`` selects the per-query score (App. D, ablation iii):
        ``projected`` uses |c_{q,i}| * ||W_O v_i - W_O o_q||, ``unprojected`` uses
        |c_{q,i}| * ||v_i - o_q||, and ``attention`` uses a_{q,i}.

        Args:
            key_states: Tensor [1, H, n, D], repeated to the query-head count.
            value_states: Tensor [1, H, n, D], repeated to the query-head count.
            query_batches: List of tensors [1, H, Q_b, D]: Q_W first, then Q_look^(1..M).
            causal_flags: True only for Q_W. The window queries must not see later window
                positions. The lookahead queries come after the prompt and need no mask.

        Returns:
            One float32 tensor [H, n] for each query set: psi(E_{i,W}), psi(E_{i,1..M}).
        """
        H = key_states.shape[1]
        n = key_states.shape[2]
        D = self.head_dim
        device = key_states.device

        value_f = value_states.squeeze(0).float()  # [H, n, D]
        key_f = key_states.squeeze(0).float()      # [H, n, D]

        out_scores = [torch.zeros(H, n, device=device, dtype=torch.float32) for _ in query_batches]

        objective = self.score_objective
        attention_only = (objective == "attention")

        for h in range(H):
            v_h = value_f[h]   # [n, D]
            k_h = key_f[h]     # [n, D]
            if attention_only:
                Wo_h = None
                Pv_h = None
                normsq_v_h = None
            else:
                # 'unprojected' is the same expression with W_O replaced by the identity.
                if objective == "projected":
                    Wo_h = self.o_proj_weight[:, h * D:(h + 1) * D].float()  # [hidden, D]
                    Pv_h = v_h @ Wo_h.T               # [n, hidden]
                else:
                    Wo_h = None
                    Pv_h = v_h                        # [n, D]
                normsq_v_h = (Pv_h * Pv_h).sum(-1)    # [n]

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
                attn_w = F.softmax(attn_logits, dim=-1)  # [Q, n]: a_{q,i}

                if attention_only:
                    # Accumulated attention mass (the H2O / SnapKV criterion) on the same
                    # queries, aggregated by the same psi.
                    for start in range(0, n, self.candidate_chunk):
                        end = min(start + self.candidate_chunk, n)
                        out_scores[b][h, start:end] = self._robust_topbeta_mean(attn_w[:, start:end])
                    continue

                y_h = attn_w @ v_h                          # [Q, D]: o_q
                Py_h = (y_h @ Wo_h.T) if Wo_h is not None else y_h   # [Q, hidden] or [Q, D]
                normsq_y_h = (Py_h * Py_h).sum(-1)           # [Q]
                c = attn_w / (1.0 - attn_w + self.causal_eps)  # [Q, n]

                for start in range(0, n, self.candidate_chunk):
                    end = min(start + self.candidate_chunk, n)
                    dot = Py_h @ Pv_h[start:end].T  # [Q, chunk]
                    dist_sq = normsq_y_h.unsqueeze(1) + normsq_v_h[start:end].unsqueeze(0) - 2 * dot
                    dist = dist_sq.clamp_min(0).sqrt()
                    out_scores[b][h, start:end] = self._robust_topbeta_mean(c[:, start:end].abs() * dist)
        return out_scores

    # ------------------------------------------------------------------
    # Budget allocation and selection (Eq. 8; Eq. 11 and App. D-C as plug-ins).
    # ------------------------------------------------------------------
    def _select_chunks(self, token_score, n, H, key_states, value_states):
        """Return the retention mask R = P u Select(S, B_h) over chunks (Eq. 8).

        The first chunk (attention sink) and the observation-window chunks are the
        protected set P. Each chunk score is divided by the mean droppable score of its
        head. Eq. 11 divides the remaining budget across heads (uniform if tau = 0).
        Top-k selection fills each head budget. If lambda_k or lambda_v is more than
        zero, greedy diversity-aware selection fills it.

        Returns:
            A bool tensor [H, n_chunk].
        """
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

        # E_{l,h}: per-head perturbation mass of the droppable chunks (Eq. 11).
        E = torch.where(drop, normalized.clamp_min(0), zeros).sum(-1)  # [H]
        total_chunks_per_head = self.base_capacity // self.chunk_size
        total_budget = H * total_chunks_per_head - H * num_preserved_per_head
        weight = (E + self.causal_eps).pow(self.budget_tau)
        raw_alloc = total_budget * weight / weight.sum()
        head_budget = raw_alloc.floor().long()
        # floor() in Eq. 11 can give a few chunks less than the layer budget. Give the
        # remainder to the heads with the largest fractional part.
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
        """Select chunks with the greedy diversity-aware rule (App. D-C).

            i* = argmax_i [S_i - lambda_k max_{j in R} cos(k_i, k_j) - lambda_v max_{j in R} cos(v_i, v_j)]

        Each chunk is represented by its mean key and mean value. The rule follows
        GraphKV (Li et al., 2025b).
        """
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
    # Entry point, called once per layer after the rollout (Algorithm 1, lines 11-24).
    # ------------------------------------------------------------------
    def compress(self):
        """Score, select and gather the compressed cache of this layer.

            S_i = psi(E_{i,W}) + lambda_L * sum_m omega_m * psi(E_{i,m})    (Eq. 7)

        Layers below ``skip`` and prompts that are not longer than the budget are not
        compressed.

        Returns:
            A tuple ``(keys, values, cache_kwargs)``. ``keys`` and ``values`` are the
            flattened per-head tensors. ``cache_kwargs`` holds the metadata for
            :class:`DynamicCacheSplitHeadFlatten`.
        """
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
        S_total = scores[0]                         # psi(E_{i,W})
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
    """Attach a :class:`TestTimeLookaheadKVCluster` to an attention module on its first call.

    The hyperparameters come from the module config. :func:`lorekv.harness.configure_model`
    writes them.
    """
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
            particle_beta=getattr(self.config, "particle_beta", 1.0),
            particle_gamma=getattr(self.config, "particle_gamma", 0.0),
            robust_alpha=self.config.robust_alpha,
            robust_topk_beta=self.config.robust_topk_beta,
            lambda_lookahead=self.config.lambda_lookahead,
            lambda_k=self.config.lambda_k,
            lambda_v=self.config.lambda_v,
            budget_tau=self.config.budget_tau,
            lookahead_temperature=self.config.lookahead_temperature,
            lambda_d=getattr(self.config, "lambda_d", 0.0),
            divergence_layer_stride=getattr(self.config, "divergence_layer_stride", 1),
            normalize_typicality=getattr(self.config, "normalize_typicality", False),
            score_objective=getattr(self.config, "score_objective", "projected"),
        )
        self.kv_cluster.o_proj_weight = self.o_proj.weight
        self.kv_cluster.head_dim = self.head_dim
        self.kv_cluster.num_key_value_groups = self.num_key_value_groups
        print(f"Compress config(Lookahead): window_size={self.kv_cluster.window_size}, "
              f"base_capacity={self.kv_cluster.base_capacity}, chunk_size={self.kv_cluster.chunk_size}, "
              f"M={self.kv_cluster.num_particles}, T={self.kv_cluster.lookahead_len}, "
              f"beta={self.kv_cluster.particle_beta}, gamma={self.kv_cluster.particle_gamma}, "
              f"lambda_d={self.kv_cluster.lambda_d}, normalize={self.kv_cluster.normalize_typicality}, "
              f"lambda_lookahead={self.kv_cluster.lambda_lookahead}, lambda_k={self.kv_cluster.lambda_k}, "
              f"lambda_v={self.kv_cluster.lambda_v}, budget_tau={self.kv_cluster.budget_tau}, "
              f"score_objective={self.kv_cluster.score_objective}", flush=True)
