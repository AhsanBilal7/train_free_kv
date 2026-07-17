"""Orchestration for "Training-Free Test-Time Lookahead Causal KV Eviction" (Algorithm 1).

This implements the multi-phase pipeline that the per-layer attention hijack
(causal_mistral_hijack.py / causal_llama_hijack.py) cannot do on its own, since it spans many
forward calls and an explicit branch point between "generate lookahead continuations" and
"commit to a single compressed cache":

  1. Prefill on the full prompt with NO eviction ("collect" phase) -- every layer's kv_cluster
     captures its observation-window queries (Eq. 2/5) and stores the full, uncompressed K/V.
  2. Roll out M short, sampled continuations from the prefill state ("lookahead" phase, Eq. 4-8),
     each on its own cloned cache so particles don't interfere; every attention layer records the
     query states produced by each particle's tokens, and we record per-particle confidence.
  3. Compute particle weights (Eq. 8) and call kv_cluster.compress() on every layer, which scores
     all candidates with the window + particle-weighted lookahead queries (Eq. 9-18), allocates a
     head-wise budget (Eq. 19-20), and runs diversity-aware greedy selection (Eq. 21-22) to build
     the final flattened, evicted cache.
  4. Discard the lookahead draft tokens (they were never appended to the real cache) and decode
     the actual answer from the compressed cache ("final" phase).
"""
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from causal_lookahead_snapkv.monkeypatch.snapkv_utils import DynamicCacheSplitHeadFlatten


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


@torch.no_grad()
def generate_with_lookahead(model, tokenizer, input_ids, attention_mask, max_gen, eos_token_ids):
    """Returns a 1D LongTensor of generated token ids (answer only, prompt excluded)."""
    device = input_ids.device
    n = input_ids.shape[1]

    cfg = model.model.config
    M = cfg.num_particles
    T = cfg.lookahead_len
    lambda_h = cfg.lambda_h
    eta = cfg.particle_eta
    temperature = cfg.lookahead_temperature

    # ---- Phase 1 (Algorithm 1, line 1): full, uncompressed prefill ----
    _set_phase(model, "collect")
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    base_cache = out.past_key_values
    last_logits = out.logits[:, -1, :].float()

    # ---- Phase 2 (Algorithm 1, lines 4-8): M sampled lookahead particles ----
    # Pre-bind every per-step local to None so the `del` below (which frees the lookahead loop's
    # cache/activation tensors before the memory-heavy compress() phase) is always valid, even if
    # M or T is configured to 0.
    cache_m = step_out = tok = next_tok = logits = logprobs = probs = ent = logps = ents = probs0 = None
    _set_phase(model, "lookahead")
    particle_scores = torch.zeros(M, device=device)
    for m in range(M):
        _set_particle(model, m)
        cache_m = _clone_dynamic_cache(base_cache)
        probs0 = F.softmax(last_logits / max(temperature, 1e-5), dim=-1)
        tok = torch.multinomial(probs0, num_samples=1)
        logps, ents = [], []
        pos = n
        for _t in range(T):
            step_out = model(
                input_ids=tok, past_key_values=cache_m, use_cache=True,
                cache_position=torch.tensor([pos], device=device),
                position_ids=torch.tensor([[pos]], device=device),
            )
            logits = step_out.logits[:, -1, :].float()
            logprobs = F.log_softmax(logits, dim=-1)
            probs = logprobs.exp()
            ent = -(probs * logprobs).sum(-1)
            next_tok = torch.multinomial(F.softmax(logits / max(temperature, 1e-5), dim=-1), num_samples=1)
            logps.append(logprobs.gather(-1, next_tok).squeeze())
            ents.append(ent.squeeze())
            cache_m = step_out.past_key_values
            tok = next_tok
            pos += 1
        # Eq. 7: R_m = mean log p - lambda_H * mean entropy.
        particle_scores[m] = torch.stack(logps).mean() - lambda_h * torch.stack(ents).mean()

    # Eq. 8: normalized particle confidence weights.
    weights = F.softmax(eta * particle_scores, dim=0)
    for layer in model.model.layers:
        layer.self_attn.kv_cluster.set_particle_weights(weights)

    # base_cache (the full, uncompressed prefill KV cache for every layer) and the lookahead loop's
    # locals (each holding a full-length cloned cache / step output) are no longer needed -- everything
    # compress() needs was already captured per-layer onto kv_cluster.full_key_states/window_queries.
    # Drop them now: compress() itself needs a sizable transient buffer per head (Eq. 9-14's W_O
    # projection), and letting these stick around as live references is what was causing the
    # CUDA OOM (torch.cuda.empty_cache() in pred.py's outer loop can't reclaim memory that's still
    # referenced by a live Python object).
    del base_cache, cache_m, step_out, tok, next_tok, logits, logprobs, probs, ent, logps, ents, probs0
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
