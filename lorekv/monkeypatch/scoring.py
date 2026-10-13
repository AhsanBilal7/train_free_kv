import math
from typing import List, Optional, Tuple

import torch
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


# ==================================================================
# Eq. 3 (reparametrized): per-step reliability statistics.
#
# The published score weighted a particle by its average log-probability and then
# subtracted lambda_H times its average predictive entropy. Those two channels are
# not independent: the particles are drawn from p_theta itself, so
#
#     E[ log p_theta(y_u | h_u) | h_u ] = -H( p_theta(. | h_u) ),
#
# i.e. the log-probability term already CONTAINS the entropy. Subtracting entropy on
# top of it does not add a second signal, it just scales the one that is already there
# -- which is why lambda_H = 0 vs 0.3 moves LongBench by 0.01 points.
#
# We therefore split the two channels explicitly. For each sampled token define the
# CENTERED score
#
#     c_u = log q(y_u | h_u) + H( q(. | h_u) ),        E[c_u | h_u] = 0,
#
# which measures only how improbable the branch actually taken was *relative to the
# uncertainty at that context* (branch atypicality), with the uncertainty itself removed.
# A trajectory is then scored as
#
#     R_m = beta * cbar_m - gamma * Hbar_m - lambda_D * Dhat_m
#
# so beta prices branch atypicality, gamma prices context uncertainty (an explicit,
# deliberate preference for low-entropy histories, matching the near-greedy deployment
# operating point), and lambda_D prices ensemble disagreement -- three independent knobs
# instead of one entropy signal counted twice.
#
# Nesting: since (1/T) sum_u log q(y_u|h_u) = cbar_m - Hbar_m, the old score is the
# (beta, gamma, lambda_D) = (1, 1 + lambda_H, lambda_D) ray. In particular the old
# "log-probability only" setting lambda_H = 0 is gamma = 1 -- a FULL unit of entropy
# penalty, not a no-penalty control. gamma = 0 was unreachable before.
#
# NOTE (centering law == sampling law): E[c_u | h_u] = 0 holds only when the particles
# are drawn from the same distribution c_u is centered against. Any truncation
# (top-k / top-p) or temperature != 1 must therefore be applied to the logits BEFORE
# the log-softmax below, otherwise q concentrates on high-probability tokens,
# E_q[log p] > -H(p), and cbar_m acquires a silent positive bias.
# ==================================================================
def mask_top_k(z, top_k):
    """-inf outside the top-k support, so log_softmax gives the truncated law exactly."""
    if not top_k or top_k <= 0 or top_k >= z.shape[-1]:
        return z
    kth = z.topk(int(top_k), dim=-1).values[..., -1:]
    return z.masked_fill(z < kth, float("-inf"))


def mask_top_p(z, top_p):
    """-inf outside the nucleus, so log_softmax gives the truncated law exactly."""
    if top_p is None or top_p >= 1.0:
        return z
    srt, idx = z.sort(dim=-1, descending=True)
    cum = srt.softmax(dim=-1).cumsum(dim=-1)
    drop = cum - srt.softmax(dim=-1) >= float(top_p)   # keep the token that crosses top_p
    drop[..., 0] = False
    return z.masked_fill(drop.scatter(-1, idx, drop), float("-inf"))


def step_stats(logits, y, temperature=1.0, top_k=None, top_p=None):
    """Per-step reliability statistics for one decoding step.

    logits : [M, |V|] raw next-token logits (the SAME logits the sampler consumed)
    y      : [M] the sampled token ids at this step

    Returns (logp, H, V, c), each [M]:
      logp -- log q(y | h)
      H    -- entropy of q(. | h)                      (nats)
      V    -- varentropy Var_{y~q}[log q(y)]           (Prop. 3)
      c    -- centered score logp + H, with E[c | h] = 0

    `temperature`, `top_k` and `top_p` MUST match the sampler's, or the centering
    law stops being the sampling law and E[c | h] != 0.
    """
    z = logits.float() / max(float(temperature), 1e-5)
    if top_k is not None:
        z = mask_top_k(z, top_k)
    if top_p is not None:
        z = mask_top_p(z, top_p)

    logq = torch.log_softmax(z, dim=-1)          # centering law == sampling law
    q = logq.exp()

    # Masked reductions: outside a truncated support q == 0 and logq == -inf, and the
    # plain product 0 * -inf is nan.
    zeros = torch.zeros_like(q)
    qlogq = torch.where(q > 0, q * logq, zeros)
    qlogq2 = torch.where(q > 0, q * logq.pow(2), zeros)

    H = -qlogq.sum(-1)                            # entropy
    V = (qlogq2.sum(-1) - H.pow(2)).clamp_min(0)  # varentropy = Var[log q]
    lp = logq.gather(-1, y.reshape(-1, 1).to(logq.device)).squeeze(-1)
    return lp, H, V, lp + H


def reliability_weights(c, H, V, D_hat, beta, gamma, lambda_d=0.0, normalize=False):
    """Eq. 3 + Eq. 6: particle weights omega from the three independent channels.

    c, H, V : [M, T] per-step centered score / entropy / varentropy
    D_hat   : [M] standardized trajectory disagreement (Eq. 5), or None
    normalize : use the variance-normalized typicality ctilde = cbar / sqrt(Vbar / T)
                instead of cbar. An ABLATED SWITCH, not a default (see test V5).

    Returns (omega [M], parts dict) so the caller can log the channels separately.
    There is no eta: softmax(eta * (beta*c - gamma*H - lambda_D*D)) is exactly
    softmax((eta*beta)*c - (eta*gamma)*H - (eta*lambda_D)*D), so eta is not
    identifiable alongside the three coefficients and has been absorbed into them.
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


# ------------------------------------------------------------------
# Eq. 7: trajectory-disagreement term of the particle confidence score.
#
# The published R_m subtracted the mean predictive entropy of a particle. That is
# redundant in expectation: the particles are themselves drawn from p_theta, so
# E[log p_theta(y_u) | y_{<u}] = -H(p_theta(.|y_{<u})), and the average log-prob term
# already carries the same low-entropy signal -- the softmax tilt then counts it twice
# and pulls the estimator toward a likelihood-tilted (mode-seeking) distribution rather
# than p_theta itself.
#
# We replace it with how ATYPICAL a particle's *future query trajectory* is relative to
# the rest of the ensemble:
#
#   D_m = 1/(M-1) sum_{r != m}  1/(T|G|) sum_{u,g} [ 1 - cos(q^{(m)}_{u,g}, q^{(r)}_{u,g}) ]
#
# where u indexes the T lookahead steps and g ranges over the (layer, head) groups G.
# Both queries in a pair sit at the same absolute position n+u, so they carry the same
# rotary phase and the cosine is directly comparable.
#
# D_m is measured in the same space the score actually consumes (query directions), and
# it is a purely ENSEMBLE-RELATIVE quantity: after the standardization below it sums to
# zero over the particles, so the tilt only re-allocates mass *within* the sample from
# p_theta instead of systematically shifting the target distribution the way an absolute
# likelihood/entropy penalty does. It also bounds the damage from a confidently-wrong
# particle: a particle whose queries nobody else reproduces is down-weighted no matter
# how confident the model was while generating it.
# ------------------------------------------------------------------
@torch.no_grad()
def divergence_from_query_stacks(stacks, num_particles, eps=1e-6, device=None):
    """Core D_m math, shared by the runner and the analysis figures.

    `stacks` yields one [M, Hq, T, D] tensor per (layer) group -- particle m's captured
    post-RoPE lookahead queries for that layer, stacked along dim 0 in particle order.
    Returns a float32 tensor [M]; all-zeros when M <= 2 or when no group supplied any
    query. The M <= 2 guard is not cosmetic: disagreement is undefined for a single
    particle, and DEGENERATE for two, because D_1 = d(1,2) = D_2 by symmetry, so
    sd(D) = 0 and the standardization in Eq. 5 would be 0/0. Any (M = 2, lambda_D > 0)
    cell therefore reduces exactly to lambda_D = 0 rather than producing NaNs.

    `device` may be left None, in which case the accumulator adopts the device of the
    first stack (the caller's query tensors normally live on the GPU already).
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
        # mean over (head, step) of the cosine between every ordered particle pair
        cos_sum += torch.einsum("mhtd,rhtd->mr", q, q) / (q.shape[1] * q.shape[2])
        n_groups += 1

    if n_groups == 0:
        return torch.zeros(M, dtype=torch.float32, device=device)
    dist = 1.0 - cos_sum / n_groups                                 # [M, M]
    dist.fill_diagonal_(0.0)                                        # exclude r == m
    return dist.sum(dim=1) / (M - 1)


def _cluster_query_stacks(layers, num_particles, layer_stride=1):
    """Yield [M, Hq, T, D] stacks straight off the per-layer lookahead buffers."""
    for layer in layers[::max(1, int(layer_stride))]:
        # getattr on the LAYER first: a hybrid backbone's linear-attention layers have
        # no `self_attn` attribute at all, so layer.self_attn raises rather than
        # returning None.
        _attn = getattr(layer, "self_attn", None)
        kvc = getattr(_attn, "kv_cluster", None) if _attn is not None else None
        if kvc is None:
            continue
        bufs = [kvc._lookahead_buffers[m] for m in range(num_particles)]
        if any(len(b) == 0 for b in bufs):
            continue
        # A particle that stopped early would break the (u, g) pairing, so compare only
        # over the steps every particle actually produced.
        T = min(len(b) for b in bufs)
        yield torch.stack([torch.cat(b[:T], dim=2)[0] for b in bufs], dim=0)


@torch.no_grad()
def particle_trajectory_divergence(layers, num_particles, layer_stride=1, eps=1e-6, device=None):
    """Mean pairwise query-trajectory disagreement D_m, one scalar per particle.

    `layers` are the decoder layers whose `self_attn.kv_cluster._lookahead_buffers[m]`
    hold the m-th particle's captured post-RoPE query states (list of T x [1,Hq,1,D]);
    they are populated by the attention hijack during the "lookahead" phase and must
    still be alive (i.e. call this BEFORE reset_lookahead_buffers()).
    """
    return divergence_from_query_stacks(
        _cluster_query_stacks(layers, int(num_particles), layer_stride),
        int(num_particles), eps=eps, device=device)


def standardize_disagreement(x, eps=1e-6):
    """Eq. 5: zero-mean / unit-std across the M particles.

    Centering is free (softmax is shift-invariant) but makes the "pure re-allocation"
    reading exact; the std normalization is what makes lambda_D scale-free, so one
    setting transfers across prompts whose absolute disagreement levels differ, and
    keeps the resulting tilt bounded (a z-score over M particles cannot exceed
    sqrt(M-1)).

    Returns all-zeros for M <= 2: disagreement is undefined for one particle and
    degenerate for two (D_1 = D_2 by symmetry, so sd(D) = 0 and the z-score is 0/0).
    Degenerate spread at larger M (every particle identical) also collapses to zeros,
    i.e. uniform weights.
    """
    if x.numel() <= 2:
        return torch.zeros_like(x)
    centered = x - x.mean()
    sd = x.std(unbiased=False)
    if float(sd) < eps:
        return torch.zeros_like(x)
    return centered / sd


# Back-compatible name used by the analysis/figure scripts.
standardize_particle_scores = standardize_disagreement


class TestTimeLookaheadKVCluster:
    """Implements "Training-Free Test-Time Lookahead Causal KV Eviction" (proposed_updated_idea.pdf).

    Unlike AnDProAdaptiveSnapKVCluster.update_kv (causal_andpro_snapkv), compression here is NOT
    triggered automatically inside the attention forward during prefill. The attention hijack only
    feeds this object raw observations (window queries, lookahead-trajectory queries, the full
    uncompressed K/V); the actual scoring + eviction happens once, in `compress()`, called explicitly
    by lorekv.monkeypatch.lookahead_runner after all M lookahead particles have run
    (Algorithm 1, lines 1-34). This split is required because the lookahead continuations themselves
    must be generated against the *full* (uncompressed) cache before eviction can happen.
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

        # Eq. 3/6: lookahead particle generation + reliability weighting.
        # R_m = beta * cbar_m - gamma * Hbar_m - lambda_D * Dhat_m, three independent
        # channels (branch atypicality / context uncertainty / ensemble disagreement).
        self.num_particles = num_particles
        self.lookahead_len = lookahead_len
        self.particle_beta = particle_beta   # beta:  typicality (centered log-prob) coefficient
        self.particle_gamma = particle_gamma # gamma: entropy coefficient
        self.lambda_d = lambda_d             # lambda_D: trajectory-disagreement coefficient
        self.normalize_typicality = bool(normalize_typicality)  # ablated switch (Prop. 3)
        self.divergence_layer_stride = divergence_layer_stride
        self.lookahead_temperature = lookahead_temperature

        # Eq. 15-18: robust aggregation + particle-weighted lookahead score.
        self.robust_alpha = robust_alpha
        self.robust_topk_beta = robust_topk_beta
        self.lambda_lookahead = lambda_lookahead

        # Eq. 19-21: adaptive head-wise budget + diversity-aware greedy selection.
        self.budget_tau = budget_tau
        self.lambda_k = lambda_k
        self.lambda_v = lambda_v

        # App. C (iii): perturbation objective. What is measured for a candidate token
        # under a scoring query, BEFORE the rho-aggregation of Eq. 15-17. The lookahead
        # queries, the reliability weights and everything downstream are untouched, so
        # this is a single-factor control on Eq. 14.
        #   'projected'   |c_{q,i}| * || W_O v_i - W_O y_q ||   (Eq. 14, the core method)
        #   'unprojected' |c_{q,i}| * ||      v_i -      y_q ||   (drops W_O only)
        #   'attention'   a_{q,i}                                (accumulated attention mass)
        self.score_objective = str(score_objective).lower()
        if self.score_objective not in ("projected", "unprojected", "attention"):
            raise ValueError(
                f"score_objective must be projected|unprojected|attention, got {score_objective!r}")

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
        # .clone(), not a view: a slice keeps the WHOLE [1, H, n, D] prefill query tensor
        # alive for every layer until compress() -- 1 GiB per layer at 128K on
        # Llama-3.1-8B, 32 GiB in total, which made RULER-131072 OOM in the lookahead
        # rollout. The values are identical; only the retained storage changes.
        self.window_queries = query_states[:, :, -w:, :].detach().clone()

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
        if q == 0:
            # An empty query batch carries no information about any candidate, so it
            # contributes zero. This is the DESIGNED behaviour, not a workaround:
            # compress() already builds `self.window_queries[:, :, :0, :]` for a
            # particle whose buffer is empty (see the q_m line in compress), i.e. a
            # zero-length batch is an expected input. What was missing is that
            # topk(k>=1) on a 0-row tensor raises "selected index k out of range", so
            # the intent was expressible but not reachable. Latent on every backbone;
            # Gemma 4 is simply the first to produce an empty buffer in practice.
            return scores.new_zeros(scores.shape[1:])
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

        `self.score_objective` selects WHAT is measured here (App. C (iii)); the queries,
        reliability weights, allocation and selection are identical in all three cases:
            'projected'   s_{q,i} = |c_{q,i}| * || W_O v_i - W_O y_q ||   (Eq. 14, default)
            'unprojected' s_{q,i} = |c_{q,i}| * ||      v_i -      y_q || (W_O -> identity)
            'attention'   s_{q,i} = a_{q,i}                               (attention mass)

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
                # 'projected' embeds the deletion effect through W_O (Eq. 14); 'unprojected'
                # is the SAME expression with W_O replaced by the identity, i.e. ||Delta o||
                # measured in head space. Only this substitution changes between the two.
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
                attn_w = F.softmax(attn_logits, dim=-1)  # [Q, n] : a_{q,i}

                if attention_only:
                    # Accumulated attention mass: the criterion H2O/SnapKV use, evaluated on
                    # the SAME window + lookahead queries and aggregated by the same rho, so
                    # only Eq. 14 is replaced.
                    for start in range(0, n, self.candidate_chunk):
                        end = min(start + self.candidate_chunk, n)
                        out_scores[b][h, start:end] = self._robust_topbeta_mean(attn_w[:, start:end])
                    continue

                y_h = attn_w @ v_h                          # [Q, D] : y_q
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
            particle_beta=getattr(self.config, "particle_beta", 1.0),
            particle_gamma=getattr(self.config, "particle_gamma", 0.0),
            robust_alpha=self.config.robust_alpha,
            robust_topk_beta=self.config.robust_topk_beta,
            lambda_lookahead=self.config.lambda_lookahead,
            lambda_k=self.config.lambda_k,
            lambda_v=self.config.lambda_v,
            budget_tau=self.config.budget_tau,
            lookahead_temperature=self.config.lookahead_temperature,
            # getattr-guarded: older configs / checkpoints predate the disagreement term,
            # and lambda_d=0 reproduces them exactly.
            lambda_d=getattr(self.config, "lambda_d", 0.0),
            divergence_layer_stride=getattr(self.config, "divergence_layer_stride", 1),
            normalize_typicality=getattr(self.config, "normalize_typicality", False),
            # App. C (iii) perturbation-objective ablation. getattr-guarded so every
            # pre-existing config keeps the Eq. 14 projected score bit-for-bit.
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
