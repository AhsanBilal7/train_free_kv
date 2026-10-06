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


def _phase_configs(model):
    """Every config object the attention modules might read the phase from.

    The attention hijack reads `self.config.lookahead_phase`, and `self.config` is
    whatever config that module was constructed with. On a multimodal backbone those
    differ: Gemma4UnifiedTextAttention holds the TEXT config while `model.model.config`
    is the top-level multimodal one. Writing only the latter left the attention
    modules permanently in "collect" -- so after eviction they kept taking the collect
    branch and called cache.update() without the head_lens/cu_klen kwargs the
    flattened layer needs (KeyError: 'head_lens'), instead of the varlen decode path.
    Set it on both, and on the layers' own config, so the read cannot miss it.
    """
    seen, out = set(), []
    for cfg in (getattr(model, "config", None),
                getattr(getattr(model, "model", None), "config", None)):
        while cfg is not None and id(cfg) not in seen:
            seen.add(id(cfg))
            out.append(cfg)
            cfg = getattr(cfg, "text_config", None)
    return out


def _set_phase(model, phase):
    for cfg in _phase_configs(model):
        cfg.lookahead_phase = phase


def _set_particle(model, idx):
    for cfg in _phase_configs(model):
        cfg.lookahead_particle_idx = idx


def _decoder_layers(model):
    """The decoder layer list, wherever this architecture keeps it.

    `model.model.layers` holds for Mistral/Llama/Qwen2 and for Qwen3.5. It does NOT
    hold for Gemma 4: Gemma4UnifiedModel is a multimodal wrapper whose decoder lives
    at `model.model.language_model.layers`. Walk the usual nestings rather than
    hardcoding one, so a future multimodal backbone does not need another special
    case here.
    """
    for path in (("model", "layers"),
                 ("model", "language_model", "layers"),
                 ("model", "text_model", "layers"),
                 ("language_model", "layers"),
                 ("layers",)):
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise AttributeError(
        f"cannot locate decoder layers on {type(model).__name__}; add its path to "
        "_decoder_layers")


def _kv_layers(model):
    """(layer_idx, kv_cluster) for every layer LORE-KV actually owns.

    On a uniform backbone that is every layer. On a hybrid one it is only the
    full-attention layers: Qwen3_5DecoderLayer has NO `self_attn` at all on a
    linear-attention layer (it builds `linear_attn` instead), so `layer.self_attn`
    raises rather than returning something falsy. Layers we never patched have no
    kv_cluster, which is the definitive test -- it also covers Gemma 4, where every
    layer does have `self_attn` but only the full-attention ones were hijacked.
    """
    for idx, layer in enumerate(_decoder_layers(model)):
        attn = getattr(layer, "self_attn", None)
        kvc = getattr(attn, "kv_cluster", None) if attn is not None else None
        if kvc is not None:
            yield idx, kvc


def _clone_dynamic_cache(cache):
    """Deep-copy the prefill cache so one lookahead particle cannot disturb another.

    Two cache layouts to support:

    transformers 4.44  `cache.key_cache[i]` / `cache.value_cache[i]` -- flat lists.

    transformers 5.x   `cache.layers[i]`, one object per layer, whose type depends
                       on the layer (DynamicLayer / DynamicSlidingWindowLayer /
                       LinearAttentionLayer). Their tensor attributes differ, so the
                       clone copies every tensor attribute it finds rather than
                       assuming `keys`/`values` -- which is REQUIRED for a hybrid
                       backbone, not just tidier: a linear-attention layer carries a
                       recurrent state and no keys/values at all, and sharing that
                       state between particles would let each rollout corrupt the
                       next. Non-tensor attributes (cumulative_length, flags) are
                       copied by value for the same reason.
    """
    if hasattr(cache, "key_cache"):                       # 4.44
        # A new cache OBJECT over the SAME tensors -- no data copy. 4.44's
        # DynamicCache.update only ever rebinds `key_cache[i] = torch.cat(...)`, which
        # allocates a fresh tensor and never writes into the old one, so a particle's
        # decode can never touch the prefill tensors shared here (the attention hijack
        # also only reads them via update()). Values are identical to a .clone(); the
        # difference is memory: a deep copy is one extra full-length cache alive per
        # particle -- 16 GiB for Llama-3.1-8B at 128K, which is what made RULER-131072
        # OOM at 78 GiB. The 5.x branch below keeps deep-copying: its recurrent layers
        # ARE updated in place.
        new_cache = DynamicCache()
        new_cache.key_cache = list(cache.key_cache)
        new_cache.value_cache = list(cache.value_cache)
        new_cache._seen_tokens = cache._seen_tokens
        return new_cache

    import copy as _copy
    new_cache = _copy.copy(cache)                         # 5.x: shallow, then rebuild layers
    new_cache.layers = []
    for layer in cache.layers:
        new_layer = _copy.copy(layer)
        for name, val in vars(layer).items():
            setattr(new_layer, name, _deep_clone(val))
        new_cache.layers.append(new_layer)
    return new_cache


def _deep_clone(val):
    """Clone a cache layer's state, whatever container transformers put it in.

    A DynamicLayer keeps plain tensors (`keys`, `values`), but a LinearAttentionLayer
    keeps **dicts keyed by state index** -- `conv_states = {0: tensor}`,
    `recurrent_states = {0: tensor}`. An earlier version of this clone tested only
    `torch.is_tensor` and `list/tuple`, so those dicts were copied by reference: all M
    lookahead particles shared ONE recurrent state and each rollout overwrote the
    previous one's. That does not crash -- it silently corrupts every particle after
    the first, which is exactly the kind of damage that shows up only as a bad score.
    """
    if torch.is_tensor(val):
        return val.clone()
    if isinstance(val, dict):
        return {k: _deep_clone(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return type(val)(_deep_clone(v) for v in val)
    return val


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
        # Drop the previous particle's cache BEFORE building the next one; otherwise
        # both are alive at once (the old one is still bound to cache_m / step_out
        # while the right-hand side runs).
        cache_m = step_out = None
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
            _decoder_layers(model), M, layer_stride=div_stride, device=device)
    d_hat = standardize_disagreement(particle_divergence)

    # Eq. 6: normalized particle reliability weights (no eta -- it is not identifiable
    # alongside (beta, gamma, lambda_D) and has been absorbed into them).
    weights, _parts = reliability_weights(
        particle_c, particle_H, particle_V, d_hat,
        beta=beta, gamma=gamma, lambda_d=lambda_d, normalize=normalize)
    for _idx, _kvc in _kv_layers(model):
        _kvc.set_particle_weights(weights)

    # base_cache (the full, uncompressed prefill KV cache for every layer) and the lookahead loop's
    # locals (each holding a full-length cloned cache / step output) are no longer needed -- everything
    # compress() needs was already captured per-layer onto kv_cluster.full_key_states/window_queries.
    # Drop them now: compress() itself needs a sizable transient buffer per head (Eq. 9-14's W_O
    # projection), and letting these stick around as live references is what was causing the
    # CUDA OOM (torch.cuda.empty_cache() in pred.py's outer loop can't reclaim memory that's still
    # referenced by a live Python object).
    # A hybrid backbone's linear-attention / sliding layers are NOT compressed, but
    # decoding still needs the state they built during prefill. base_cache is about to
    # be freed, so keep a reference to exactly those layer objects first -- dropping
    # them would silently reset 24 of Qwen3.5's 32 layers to empty and produce fluent
    # nonsense rather than an error.
    _owned = {i for i, _ in _kv_layers(model)}
    _carry = None
    if hasattr(base_cache, "layers"):
        _carry = [(i, L) for i, L in enumerate(base_cache.layers) if i not in _owned]

    del base_cache, cache_m, step_out, tok, next_tok, logits, q0, cs, hs, vs
    torch.cuda.empty_cache()

    # ---- Phase 3 (Algorithm 1, lines 9-34): score + evict, once per layer ----
    if _carry is None:
        compressed_cache = DynamicCacheSplitHeadFlatten()          # 4.44: every layer is ours
    else:
        from causal_lookahead_snapkv.monkeypatch.causal_lookahead_hybrid_hijack import (
            make_hybrid_flat_cache)
        compressed_cache = make_hybrid_flat_cache(model.config)
        for _i, _L in _carry:                                       # restore untouched layers
            compressed_cache.layers[_i] = _L
    for layer_idx, kv_cluster in _kv_layers(model):
        k_flat, v_flat, cache_kwargs = kv_cluster.compress()
        compressed_cache.update(k_flat, v_flat, layer_idx, cache_kwargs)
        kv_cluster.reset_lookahead_buffers()
        kv_cluster.release_collected_state()
    torch.cuda.empty_cache()

    _set_phase(model, "final")

    # ---- Phase 4 (Algorithm 1, line 35-36 + real decoding): discard lookahead drafts, answer
    # generation continues from the *real* next token (the one sampled right after prefill,
    # before any lookahead branching) using the now-compressed cache.
    # 4.44 threads a legacy tuple back through the patched model forward; the 5.x
    # hybrid path keeps the Cache object (its model forward takes past_key_values
    # directly and make_hybrid_model_forward already marked this one as flat).
    past_key_values = (compressed_cache.to_legacy_cache()
                       if hasattr(compressed_cache, "to_legacy_cache") and _carry is None
                       else compressed_cache)
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
