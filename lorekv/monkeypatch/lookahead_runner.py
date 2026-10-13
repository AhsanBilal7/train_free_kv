"""LORE-KV generation for one prompt (Algorithm 1).

:func:`generate_with_lookahead` runs the method around the per-layer state in
:class:`lorekv.monkeypatch.scoring.TestTimeLookaheadKVCluster`. The patched
attention forwards read ``config.lookahead_phase``:

1. ``collect``: The model reads the full prompt without eviction. Each layer
   records Q_W and the full K and V (Algorithm 1, lines 1-2).
2. ``lookahead``: The model samples M continuations of T tokens (Eq. 1). Each
   continuation uses its own copy of the prefill cache. Each layer records the
   queries of each continuation. The runner keeps c_u, H_u and V_u of each
   sampled token (Eqs. 2 and 9; lines 3-6).
3. The runner computes the weights omega (Eqs. 3-4, 10; lines 8-10). It then
   calls ``compress()`` on each layer (Eqs. 5-8; lines 11-24).
4. ``final``: The runner discards the continuations and decodes the answer
   greedily from the compressed cache (lines 26-27).

The runner samples the first token of each continuation from the prefill logits.
The statistics of step u belong to the token sampled at step u.
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
    """Return each config object that an attention module can read the phase from.

    On a multimodal backbone (Gemma 4) the attention modules read the nested text
    config. The phase is thus written to each config in the chain.
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
    """Return the list of decoder layers of the model.

    The list is ``model.model.layers`` for Mistral, Llama, Qwen2 and Qwen3.5, and
    ``model.model.language_model.layers`` for the multimodal Gemma 4 model.
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
    """Yield ``(layer_idx, kv_cluster)`` for each layer that LORE-KV compresses.

    On a dense backbone this is each layer. On a hybrid backbone it is each
    full-attention layer (paper App. E.2). A layer is compressed if its attention
    module has a ``kv_cluster``.
    """
    for idx, layer in enumerate(_decoder_layers(model)):
        attn = getattr(layer, "self_attn", None)
        kvc = getattr(attn, "kv_cluster", None) if attn is not None else None
        if kvc is not None:
            yield idx, kvc


def _clone_dynamic_cache(cache):
    """Make a copy of the prefill cache for one continuation.

    transformers 4.44: The copy is a new cache object over the same tensors.
    ``DynamicCache.update`` makes a new tensor with ``torch.cat`` and does not write to
    the old tensor, so the tensors can be shared. This saves one full cache for each
    continuation (16 GiB for Llama-3.1-8B at 128K tokens).

    transformers 5.x: The function clones each tensor of each layer, because the
    linear-attention layers change their recurrent state in place.
    """
    if hasattr(cache, "key_cache"):                       # 4.44
        # A new cache object over the same tensors (no copy). DynamicCache.update
        # replaces a layer with a new torch.cat result and does not write to the old
        # tensor. The prefill tensors thus stay unchanged.
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
    """Clone a value of a cache layer. Tensors in dicts, lists and tuples are also cloned."""
    if torch.is_tensor(val):
        return val.clone()
    if isinstance(val, dict):
        return {k: _deep_clone(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return type(val)(_deep_clone(v) for v in val)
    return val


def _shape_logits(logits, temperature, top_k=None, top_p=None):
    """Apply the temperature and truncation of the sampler to the logits.

    :func:`lorekv.monkeypatch.scoring.step_stats` centers the statistics on the same
    distribution, so E[c_u | h_u] = 0 (Eq. 2).
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
    """Run Algorithm 1 on one prompt and return the token IDs of the answer.

    The answer is decoded greedily, as in ``model.generate`` for the baselines.

    Args:
        model: A backbone patched with ``replace_*_lookahead`` and configured with
            :func:`lorekv.harness.configure_model`. The batch size is 1.
        tokenizer: The tokenizer of the model.
        input_ids: Tensor [1, n] with the prompt token IDs.
        attention_mask: Tensor [1, n].
        max_gen: Maximum number of answer tokens.
        eos_token_ids: Token IDs that stop decoding.

    Returns:
        A 1-D tensor with the answer token IDs. The prompt is not included.
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
        # Release the cache of the previous trajectory before the next copy is made.
        cache_m = step_out = None
        cache_m = _clone_dynamic_cache(base_cache)
        # Sample the first token from the same distribution that centers the statistics.
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
    # Compute D_m (Eq. 10) before compress() and reset_lookahead_buffers() clear the
    # lookahead query buffers.
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

    # Each kv_cluster holds the data that compress() needs. Release the prefill cache
    # and the loop tensors before scoring. Keep the layers that are not compressed
    # (linear-attention and sliding-window layers of a hybrid backbone), because
    # decoding needs their prefill state.
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
    # The first answer token is the argmax of the prefill logits, as with the full cache.
    # transformers 4.44 uses a legacy tuple cache; the 5.x hybrid path uses the Cache object.
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
