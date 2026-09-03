"""Orchestration for "Training-Free Test-Time Lookahead Causal KV Eviction" (Algorithm 1).

This implements the multi-phase pipeline that the per-layer attention hijack
(causal_mistral_hijack.py / causal_llama_hijack.py) cannot do on its own, since it spans many
forward calls and an explicit branch point between "generate lookahead continuations" and
"commit to a single compressed cache":

  1. Prefill on the full prompt with NO eviction ("collect" phase) -- every layer's kv_cluster
     captures its observation-window queries (Eq. 2/5) and stores the full, uncompressed K/V.
  2. Roll out M short, sampled continuations from the prefill state ("lookahead" phase, Eq. 1),
     each on its own cloned cache so particles don't interfere; every attention layer records the
     query states produced by each particle's tokens, and we record the per-STEP reliability
     statistics (centered score c_u, entropy H_u, varentropy V_u) of each particle.
  3. Compute particle weights (Eq. 3/6) and call kv_cluster.compress() on every layer, which scores
     all candidates with the window + particle-weighted lookahead queries (Eq. 9-18), allocates a
     head-wise budget (Eq. 19-20), and runs diversity-aware greedy selection (Eq. 21-22) to build
     the final flattened, evicted cache.
  4. Discard the lookahead draft tokens (they were never appended to the real cache) and decode
     the actual answer from the compressed cache ("final" phase).
"""
import torch
from transformers.cache_utils import DynamicCache

from causal_lookahead_snapkv.monkeypatch.snapkv_utils import (
    DynamicCacheSplitHeadFlatten,
    particle_trajectory_divergence,
    reliability_weights,
    standardize_disagreement,
    step_stats,
)


def _set_phase(model, phase):
    model.model.config.lookahead_phase = phase


def _set_particle(model, idx):
    model.model.config.lookahead_particle_idx = idx


def _clone_dynamic_cache(cache):
    new_cache = DynamicCache()
    for layer_idx in range(len(cache.key_cache)):
        new_cache.key_cache.append(cache.key_cache[layer_idx].clone())
        new_cache.value_cache.append(cache.value_cache[layer_idx].clone())
    new_cache._seen_tokens = cache._seen_tokens
    return new_cache


def _shape_logits(logits, temperature, top_k=None, top_p=None):
    """Apply the sampler's temperature/truncation. The particle statistics are centered
    against exactly this law (see snapkv_utils.step_stats), so the two must never drift
    apart -- both call sites derive their distribution from this one function."""
    from causal_lookahead_snapkv.monkeypatch.snapkv_utils import mask_top_k, mask_top_p
    z = logits.float() / max(float(temperature), 1e-5)
    if top_k is not None:
        z = mask_top_k(z, top_k)
    if top_p is not None:
        z = mask_top_p(z, top_p)
    return z


@torch.no_grad()
def generate_with_lookahead(model, tokenizer, input_ids, attention_mask, max_gen, eos_token_ids):
    """Returns a 1D LongTensor of generated token ids (answer only, prompt excluded)."""
    device = input_ids.device
    n = input_ids.shape[1]

    cfg = model.model.config
    M = cfg.num_particles
    T = cfg.lookahead_len
    # Eq. 3: R_m = beta * cbar_m - gamma * Hbar_m - lambda_D * Dhat_m.
    beta = getattr(cfg, "particle_beta", 1.0)
    gamma = getattr(cfg, "particle_gamma", 0.0)
    lambda_d = getattr(cfg, "lambda_d", 0.0)
    normalize = bool(getattr(cfg, "normalize_typicality", False))
    div_stride = getattr(cfg, "divergence_layer_stride", 1)
    temperature = cfg.lookahead_temperature
    top_k = getattr(cfg, "lookahead_top_k", None)
    top_p = getattr(cfg, "lookahead_top_p", None)

    # ---- Phase 1 (Algorithm 1, line 1): full, uncompressed prefill ----
    _set_phase(model, "collect")
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    base_cache = out.past_key_values
    last_logits = out.logits[:, -1, :].float()

    # ---- Phase 2 (Algorithm 1, lines 4-8): M sampled lookahead particles ----
    # Pre-bind every per-step local to None so the `del` below (which frees the lookahead loop's
    # cache/activation tensors before the memory-heavy compress() phase) is always valid, even if
    # M or T is configured to 0.
    cache_m = step_out = tok = next_tok = logits = q0 = None
    cs = hs = vs = None
    _set_phase(model, "lookahead")
    # Per-STEP statistics, not only trajectory means: the validation diagnostics
    # (autocorrelation of c_u, the Prop. 3 variance regression, the V5 scale ratio)
    # all need the sequence, and it is only M x T floats per prompt.
    particle_c = torch.zeros(M, T, device=device)     # centered score c_u = log q + H
    particle_H = torch.zeros(M, T, device=device)     # entropy H_u
    particle_V = torch.zeros(M, T, device=device)     # varentropy V_u
    for m in range(M):
        _set_particle(model, m)
        cache_m = _clone_dynamic_cache(base_cache)
        # First token of the particle: sampled from the SAME truncated/tempered law the
        # per-step statistics below are centered against.
        q0 = torch.log_softmax(_shape_logits(last_logits, temperature, top_k, top_p), dim=-1).exp()
        tok = torch.multinomial(q0, num_samples=1)
        cs, hs, vs = [], [], []
        pos = n
        for _t in range(T):
            step_out = model(
                input_ids=tok, past_key_values=cache_m, use_cache=True,
                cache_position=torch.tensor([pos], device=device),
                position_ids=torch.tensor([[pos]], device=device),
            )
            logits = step_out.logits[:, -1, :].float()
            z = _shape_logits(logits, temperature, top_k, top_p)
            next_tok = torch.multinomial(torch.softmax(z, dim=-1), num_samples=1)
            # step_stats re-applies the same shaping, so the centering law IS the
            # sampling law and E[c_u | h_u] = 0 holds exactly (paper Sec. 2.2).
            _, h_u, v_u, c_u = step_stats(logits, next_tok.squeeze(-1),
                                          temperature=temperature, top_k=top_k, top_p=top_p)
            cs.append(c_u.squeeze())
            hs.append(h_u.squeeze())
            vs.append(v_u.squeeze())
            cache_m = step_out.past_key_values
            tok = next_tok
            pos += 1
        particle_c[m] = torch.stack(cs)
        particle_H[m] = torch.stack(hs)
        particle_V[m] = torch.stack(vs)

    # Eq. 3: R_m = beta * cbar_m - gamma * Hbar_m - lambda_D * Dhat_m.
    #
    # Three independent channels:
    #   beta     branch atypicality -- did this particle take steps the model itself
    #            deemed improbable at the contexts it visited? c_u = log q + H has
    #            conditional expectation zero, so this channel carries no entropy.
    #   gamma    context uncertainty -- an explicit preference for low-entropy
    #            histories, matching the near-greedy deployment operating point.
    #   lambda_D ensemble disagreement -- how far particle m's future-query trajectory
    #            sits from the rest of the ensemble (particle_trajectory_divergence).
    #
    # Reproducing the published (lambda_H, eta) runs:
    #   beta = eta, gamma = eta * (1 + lambda_H_old), lambda_D = eta * lambda_D_old.
    # In particular the old "log-probability only" setting (lambda_H = 0) is gamma = 1,
    # i.e. a FULL unit of entropy penalty -- it was never a no-penalty control.
    # Uniform weights are (beta, gamma, lambda_D) = (0, 0, 0).
    #
    # This must run before compress()/reset_lookahead_buffers() below, which is what
    # frees the per-layer lookahead query buffers D_m is computed from.
    particle_divergence = torch.zeros(M, device=device)
    if lambda_d != 0.0:
        particle_divergence = particle_trajectory_divergence(
            model.model.layers, M, layer_stride=div_stride, device=device)
    d_hat = standardize_disagreement(particle_divergence)

    # Eq. 6: normalized particle reliability weights (no eta -- it is not identifiable
    # alongside (beta, gamma, lambda_D) and has been absorbed into them).
    weights, _parts = reliability_weights(
        particle_c, particle_H, particle_V, d_hat,
        beta=beta, gamma=gamma, lambda_d=lambda_d, normalize=normalize)
    for layer in model.model.layers:
        layer.self_attn.kv_cluster.set_particle_weights(weights)

    # base_cache (the full, uncompressed prefill KV cache for every layer) and the lookahead loop's
    # locals (each holding a full-length cloned cache / step output) are no longer needed -- everything
    # compress() needs was already captured per-layer onto kv_cluster.full_key_states/window_queries.
    # Drop them now: compress() itself needs a sizable transient buffer per head (Eq. 9-14's W_O
    # projection), and letting these stick around as live references is what was causing the
    # CUDA OOM (torch.cuda.empty_cache() in pred.py's outer loop can't reclaim memory that's still
    # referenced by a live Python object).
    del base_cache, cache_m, step_out, tok, next_tok, logits, q0, cs, hs, vs
    torch.cuda.empty_cache()

    # ---- Phase 3 (Algorithm 1, lines 9-34): score + evict, once per layer ----
    compressed_cache = DynamicCacheSplitHeadFlatten()
    for layer_idx, layer in enumerate(model.model.layers):
        kv_cluster = layer.self_attn.kv_cluster
        k_flat, v_flat, cache_kwargs = kv_cluster.compress()
        compressed_cache.update(k_flat, v_flat, layer_idx, cache_kwargs)
        kv_cluster.reset_lookahead_buffers()
        kv_cluster.release_collected_state()
    torch.cuda.empty_cache()

    _set_phase(model, "final")

    # ---- Phase 4 (Algorithm 1, line 35-36 + real decoding): discard lookahead drafts, answer
    # generation continues from the *real* next token (the one sampled right after prefill,
    # before any lookahead branching) using the now-compressed cache.
    past_key_values = compressed_cache.to_legacy_cache()
    next_token = last_logits.argmax(dim=-1, keepdim=True)
    eos_set = set(eos_token_ids)
    if int(next_token.item()) in eos_set:
        return next_token.new_zeros((0,))
    generated = [next_token]
    pos = n
    for _ in range(max_gen - 1):
        cache_position = torch.tensor([pos], device=device)
        step_out = model(
            input_ids=next_token, past_key_values=past_key_values, use_cache=True,
            cache_position=cache_position, position_ids=cache_position.unsqueeze(0),
        )
        past_key_values = step_out.past_key_values
        logits = step_out.logits[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)
        if int(next_token.item()) in eos_set:
            break
        generated.append(next_token)
        pos += 1

    return torch.cat(generated, dim=1)[0]
