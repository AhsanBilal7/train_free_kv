"""🔭 LORE-KV generation for one prompt: lookahead, reliability weighting, eviction, decoding.

:func:`generate_with_lookahead` implements Algorithm 1 of the paper around the
per-layer state in :class:`lorekv.monkeypatch.scoring.TestTimeLookaheadKVCluster`.
The patched attention forwards read ``config.lookahead_phase`` to decide what to do:

1. ``"collect"`` -- full prefill of the prompt with no eviction. Every layer records
   its observation-window queries Q_W and the full K/V (Algorithm 1, lines 1-2).
2. ``"lookahead"`` -- M continuations of T tokens are sampled from the frozen model
   (Eq. 1), each on its own copy of the prefill cache. Every layer records the
   query states of each trajectory, and the per-step statistics c_u, H_u, V_u of
   each sampled token are kept (Eqs. 2, 9; lines 3-6).
3. Trajectory weights omega (Eqs. 3-4, 10; lines 8-10), then ``compress()`` on every
   layer scores, allocates and selects (Eqs. 5-8; lines 11-24).
4. ``"final"`` -- the continuations are discarded (they were never appended to the
   real cache) and the answer is decoded greedily from the compressed cache with
   varlen flash attention (lines 26-27).

The first token of a trajectory is sampled from the prefill logits; the
statistics of step u belong to the token sampled at that step.
"""
import torch
from transformers.cache_utils import DynamicCache

from lorekv.monkeypatch.scoring import (
    DynamicCacheSplitHeadFlatten,
    particle_trajectory_divergence,
    reliability_weights,
    standardize_disagreement,
    step_stats,
)


def _phase_configs(model):
    """Every config object an attention module may read the phase from.

    On a multimodal backbone (Gemma 4) the attention modules hold the nested text
    config, not ``model.model.config``, so the phase is written to the whole chain.
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

    ``model.model.layers`` for Mistral, Llama, Qwen2 and Qwen3.5;
    ``model.model.language_model.layers`` for the multimodal Gemma 4 wrapper.
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
    """(layer_idx, kv_cluster) for every layer LORE-KV compresses.

    Every layer on a dense backbone; only the full-attention layers on a hybrid one
    (App. E.2). A layer is ours iff its attention module carries a ``kv_cluster``.
    """
    for idx, layer in enumerate(_decoder_layers(model)):
        attn = getattr(layer, "self_attn", None)
        kvc = getattr(attn, "kv_cluster", None) if attn is not None else None
        if kvc is not None:
            yield idx, kvc


def _clone_dynamic_cache(cache):
    """A per-trajectory copy of the prefill cache, so trajectories cannot interfere.

    transformers 4.44 (``key_cache`` / ``value_cache`` lists): a new cache object over
    the same tensors. ``DynamicCache.update`` only ever rebinds a layer to a new
    ``torch.cat`` result, so sharing is safe and saves one full-length cache per
    trajectory (16 GiB for Llama-3.1-8B at 128K).

    transformers 5.x (``cache.layers``): every tensor attribute of every layer is
    cloned, because linear-attention layers update their recurrent state in place.
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
    """Clone a cache layer's state through tensors, dicts, lists and tuples.

    LinearAttentionLayer keeps its conv / recurrent states in dicts keyed by state
    index; copying those by reference would let trajectories share one state.
    """
    if torch.is_tensor(val):
        return val.clone()
    if isinstance(val, dict):
        return {k: _deep_clone(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return type(val)(_deep_clone(v) for v in val)
    return val


def _shape_logits(logits, temperature, top_k=None, top_p=None):
    """Apply the sampler's temperature and truncation.

    The trajectory statistics are centered against exactly this distribution (see
    :func:`lorekv.monkeypatch.scoring.step_stats`), which keeps Eq. 2 zero-mean.
    """
    from lorekv.monkeypatch.scoring import mask_top_k, mask_top_p
    z = logits.float() / max(float(temperature), 1e-5)
    if top_k is not None:
        z = mask_top_k(z, top_k)
    if top_p is not None:
        z = mask_top_p(z, top_p)
    return z


@torch.no_grad()
def generate_with_lookahead(model, tokenizer, input_ids, attention_mask, max_gen, eos_token_ids):
    """Run Algorithm 1 for one prompt and return the answer's token ids (1D, prompt excluded).

    model          a backbone patched by ``replace_*_lookahead`` and configured by
                   :func:`lorekv.harness.configure_model` (batch size 1)
    input_ids      [1, n] prompt token ids; attention_mask the matching mask
    max_gen        maximum number of answer tokens
    eos_token_ids  token ids that stop decoding

    The answer is decoded greedily, exactly as the baselines' ``model.generate``.
    """
    device = input_ids.device
    n = input_ids.shape[1]

    cfg = model.model.config
    M = cfg.num_particles
    T = cfg.lookahead_len
    # Reliability coefficients of Eq. 3.
    beta = getattr(cfg, "particle_beta", 1.0)
    gamma = getattr(cfg, "particle_gamma", 0.0)
    lambda_d = getattr(cfg, "lambda_d", 0.0)
    normalize = bool(getattr(cfg, "normalize_typicality", False))
    div_stride = getattr(cfg, "divergence_layer_stride", 1)
    temperature = cfg.lookahead_temperature
    top_k = getattr(cfg, "lookahead_top_k", None)
    top_p = getattr(cfg, "lookahead_top_p", None)

    # ---- Phase 1 (Algorithm 1, lines 1-2): full, uncompressed prefill ----
    _set_phase(model, "collect")
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    base_cache = out.past_key_values
    last_logits = out.logits[:, -1, :].float()

    # ---- Phase 2 (Algorithm 1, lines 3-6): M sampled lookahead trajectories (Eq. 1) ----
    # Pre-bind the loop locals so the `del` before compress() is valid even for M = 0.
    cache_m = step_out = tok = next_tok = logits = q0 = None
    cs = hs = vs = None
    _set_phase(model, "lookahead")
    # Per-step statistics (M x T floats); the normalized typicality of App. C.3 needs V_u.
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
            # step_stats re-applies the same shaping, so E[c_u | h_u] = 0 (Eq. 2).
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

    # ---- Algorithm 1, lines 8-10: trajectory weights ----
    # D_m (Eq. 10) is read from the per-layer lookahead query buffers, so it must be
    # computed before compress() / reset_lookahead_buffers() free them.
    particle_divergence = torch.zeros(M, device=device)
    if lambda_d != 0.0:
        particle_divergence = particle_trajectory_divergence(
            _decoder_layers(model), M, layer_stride=div_stride, device=device)
    d_hat = standardize_disagreement(particle_divergence)

    # omega = softmax(R) (Eqs. 3-4).
    weights, _parts = reliability_weights(
        particle_c, particle_H, particle_V, d_hat,
        beta=beta, gamma=gamma, lambda_d=lambda_d, normalize=normalize)
    for _idx, _kvc in _kv_layers(model):
        _kvc.set_particle_weights(weights)

    # Everything compress() needs is already stored on each layer's kv_cluster, so free
    # the prefill cache and the rollout's locals before the memory-heavy scoring.
    # A hybrid backbone's linear-attention / sliding layers are not compressed but
    # their prefill state is still needed for decoding, so keep those layer objects.
    _owned = {i for i, _ in _kv_layers(model)}
    _carry = None
    if hasattr(base_cache, "layers"):
        _carry = [(i, L) for i, L in enumerate(base_cache.layers) if i not in _owned]

    del base_cache, cache_m, step_out, tok, next_tok, logits, q0, cs, hs, vs
    torch.cuda.empty_cache()

    # ---- Phase 3 (Algorithm 1, lines 11-24): score, allocate, select, gather ----
    if _carry is None:
        compressed_cache = DynamicCacheSplitHeadFlatten()          # 4.44: every layer is ours
    else:
        from lorekv.monkeypatch.causal_lookahead_hybrid_hijack import (
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

    # ---- Phase 4 (Algorithm 1, lines 26-27): discard the trajectories, decode greedily ----
    # The first answer token is the argmax of the prefill logits, as for the full cache.
    # transformers 4.44 threads a legacy tuple through the patched model forward; the
    # 5.x hybrid path keeps the Cache object.
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
