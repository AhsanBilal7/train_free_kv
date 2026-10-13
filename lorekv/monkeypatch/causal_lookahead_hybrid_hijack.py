"""LORE-KV attention hijacks for hybrid-attention backbones on transformers 5.x.

Why this file exists separately from causal_llama_hijack.py / _qwen2_hijack.py:
those target the transformers 4.44 `*FlashAttention2.forward` signature, and 4.48
deleted that class outright. Qwen3.5 (`qwen3_5`) and Gemma 4 (`gemma4_unified`)
only exist from transformers 5.x, whose attention interface is different -- and,
usefully, simpler:

    forward(hidden_states, position_embeddings, attention_mask, past_key_values, **kw)

`position_embeddings` arrives precomputed as (cos, sin) from the model, which
removes the 4.44 trap where a table-based rotary had to be sized from
position_ids rather than the cache (the flattened post-eviction cache reports a
length of 1 while positions keep counting). In 5.x the model does that once and
hands it down, so the bug cannot occur here.

WHICH LAYERS THIS TOUCHES
-------------------------
Only layers whose KV cache grows with context -- see hybrid_support.py for why
that is the right scope and not a limitation. The two models make it easy in
different ways:

  Qwen3.5   `Qwen3_5DecoderLayer.__init__` builds a `Qwen3_5Attention` ONLY for
            full_attention layers; linear layers get `Qwen3_5GatedDeltaNet`.
            Patching Qwen3_5Attention therefore scopes us automatically, and the
            gated-delta-net recurrent state is never reachable from our code.

  Gemma 4   ONE class (`Gemma4UnifiedTextAttention`) serves both layer types,
            distinguished by `self.layer_type`. The check must be explicit, and
            getting it wrong would silently compress sliding layers -- which is a
            different experiment (evicting a 1024-entry window to B=128 is not
            evicting a 64k cache to B=128) and must not be averaged in.

Both hijacks fall through to the stock forward for any layer we do not own, so a
mistake fails closed (uncompressed) rather than silently wrong.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from lorekv.monkeypatch.scoring import init_lookahead_snapkv
from lorekv.monkeypatch.hybrid_support import (
    compressible_layers, layer_is_compressible, resolve_budget)


_VARLEN_FALLBACK_WARNED = False


def _varlen_attend(q, k, v, cu_qlen, cu_klen, max_seqlen_k, softmax_scale=None):
    """Ragged single-token attention: flash_attn if usable, else a torch fallback.

    The decode step attends ONE query per head against that head's own slice of the
    flattened cache, and head slices have different lengths. flash_attn_varlen_func
    is the fast path. It is not always available here -- prebuilt wheels need
    GLIBC >= 2.32 and ACES has 2.28, so flash-attn must be compiled from source --
    and a missing kernel should not silently become a wrong answer or a hard stop.

    The fallback pads the per-head slices into [H, max_len, D] and masks the padding
    to -inf. That is exact, not approximate: with q_len == 1 every query attends to
    its whole (already causal) slice, so there is no causal mask to reconstruct.
    Slower than the kernel, but it makes correctness independent of the build.
    """
    try:
        import flash_attn
        # An import-only stub may be installed so the 4.44 hijacks' module-scope
        # imports resolve on this venv (see site-packages/flash_attn/__init__.py).
        # Importing it must NOT be read as "the kernel is available".
        flash_attn_varlen_func = (None if getattr(flash_attn, "__lorekv_stub__", False)
                                  else flash_attn.flash_attn_varlen_func)
        # flash-attn is CUDA-only. scripts/harness_hybrid_cycle.py drives this exact
        # path on CPU, so dispatch on the tensor's device rather than on import
        # success -- otherwise installing the real kernel silently breaks the fast
        # local test loop.
        if not q.is_cuda:
            flash_attn_varlen_func = None
    except Exception:
        flash_attn_varlen_func = None

    if flash_attn_varlen_func is not None:
        try:
            return flash_attn_varlen_func(q, k, v, cu_qlen, cu_klen, 1, max_seqlen_k,
                                          causal=True, softmax_scale=softmax_scale)
        except RuntimeError as exc:
            # The kernel's head-dim ceiling depends on the build, the dtype and the
            # sequence extents, so "head dimension at most 256" can fire for a
            # head_dim that works in isolation at other shapes. Falling back is
            # strictly better than failing the run: the torch path below is exact
            # (verified to 1.2e-07 against a per-head reference), just slower.
            if "head dimension" not in str(exc):
                raise
            global _VARLEN_FALLBACK_WARNED
            if not _VARLEN_FALLBACK_WARNED:
                print("[lorekv] flash_attn_varlen rejected head_dim=%d (%s); "
                      "using the exact torch fallback for the ragged decode."
                      % (q.shape[-1], str(exc)[:60]), flush=True)
                _VARLEN_FALLBACK_WARNED = True

    h = cu_qlen.numel() - 1
    d = q.shape[-1]
    starts = cu_klen[:-1].tolist()
    ends = cu_klen[1:].tolist()
    lens = [e - s for s, e in zip(starts, ends)]
    m = max(lens)
    kf, vf = k.view(-1, d), v.view(-1, d)
    kp = kf.new_zeros((h, m, d))
    vp = vf.new_zeros((h, m, d))
    mask = torch.zeros((h, m), dtype=torch.bool, device=q.device)
    for i, (s, ln) in enumerate(zip(starts, lens)):
        kp[i, :ln] = kf[s:s + ln]
        vp[i, :ln] = vf[s:s + ln]
        mask[i, :ln] = True
    qh = q.view(h, 1, d).float()
    # The module's OWN scaling, never a hardcoded 1/sqrt(d). Gemma 4 sets
    # `self.scaling = 1.0` because the scale is folded into q_norm; dividing again
    # here shrank every logit by ~sqrt(512) = 22.6, which flattens the softmax to
    # near-uniform attention. That does not crash -- it produces fluent-looking
    # token spew ("_____0_11__"), which is exactly what the Gemma 4 runs emitted
    # while Qwen3.5 (scaling = 1/sqrt(d), so the hardcoded value happened to match)
    # stayed clean.
    scale = (d ** -0.5) if softmax_scale is None else float(softmax_scale)
    logits = torch.bmm(qh, kp.float().transpose(1, 2)) * scale        # [H,1,m]
    logits = logits.masked_fill(~mask.unsqueeze(1), float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    out = torch.bmm(probs, vp.float())                                # [H,1,d]
    return out.to(q.dtype).view(h, 1, d)


def _flat_append(flat, new_rows, head_lens, cu_klen):
    """Append one row per head to a ragged [sum_h len_h, D] buffer.

    tiny_api_cuda.update_flatten_view does this in one CUDA pass and is the fast
    path. It is not always importable -- it is a compiled extension that needs
    `module load GCC/11.3.0` for GLIBCXX_3.4.29 at RUN time, not just build time,
    and it is CUDA-only so it cannot serve a CPU debugging run at all.

    The fallback builds the new buffer with a scatter: each head's slice grows by
    one, so destination offsets shift by the head's index. Exact, just slower.
    """
    try:
        from tiny_api_cuda import update_flatten_view
        return update_flatten_view(flat, new_rows, head_lens, cu_klen)
    except Exception:
        pass
    h = int(head_lens.numel())
    lens = head_lens.tolist()
    starts = cu_klen[:h].tolist()
    out = flat.new_empty((flat.shape[0] + h, flat.shape[-1]))
    w = 0
    for i in range(h):
        ln = int(lens[i])
        s0 = int(starts[i])
        out[w:w + ln] = flat[s0:s0 + ln]
        out[w + ln] = new_rows[i]
        w += ln + 1
    return out


def make_hybrid_flat_cache(config, past_key_values=None):
    """Cache for the post-eviction phase of a hybrid model.

    transformers 5.x builds `DynamicCache(config=...)` as a LIST of per-layer cache
    objects chosen by layer type (DynamicLayer / DynamicSlidingWindowLayer /
    LinearAttentionLayer). That is exactly the structure this needs: we replace the
    layer objects at the compressed indices with a ragged per-head layer and leave
    every other layer's native implementation untouched.

    This is what makes the hybrid port tractable. On 4.44 the whole cache had to be
    swapped for DynamicCacheSplitHeadFlatten, which assumes every layer is a plain
    growing KV cache -- an assumption a linear-attention or sliding layer breaks.
    Here the substitution is per layer, so the gated-delta-net recurrent state and
    the sliding windows keep their own (correct, already-bounded) behaviour while
    the full-attention layers get the flattened layout eviction needs.
    """
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache(config=config)
    for idx in compressible_layers(config):
        if idx < len(cache.layers):
            cache.layers[idx] = _SplitHeadFlattenLayer()
    cache._lorekv_hybrid_flat = True
    if past_key_values is not None:
        # Carry over what the collect phase already wrote for the layers we did not
        # replace; the compressed layers are re-seeded by compress() itself.
        for idx, layer in enumerate(getattr(past_key_values, "layers", [])):
            if idx < len(cache.layers) and not isinstance(cache.layers[idx], _SplitHeadFlattenLayer):
                cache.layers[idx] = layer
    return cache


class _SplitHeadFlattenLayer:
    """One compressed layer's cache: heads concatenated into a single ragged run.

    After Eq. 19-20 each head of this layer keeps a different number of tokens, so
    there is no rectangular [B,H,S,D] tensor to store. Keys/values are held flat as
    [sum_h len_h, D] and served by flash_attn_varlen_func. `update` appends exactly
    one token per head, which is what tiny_api_cuda.update_flatten_view does in a
    single pass rather than H slice-and-concat operations.

    Deliberately duck-typed rather than subclassing DynamicLayer: the base class
    assumes `self.keys.shape[-2]` is a sequence length, which is meaningless for a
    flattened layout, and inheriting it would let that assumption leak back in.
    """

    # The full CacheLayerMixin/DynamicLayer surface transformers may touch. Declared
    # up front from that interface rather than discovered one AttributeError at a
    # time; the methods that would be WRONG for a ragged layout (get_seq_length,
    # get_mask_sizes, crop, batch_*) are overridden or refused below.
    is_sliding = False
    is_initialized = True
    is_compileable = False
    supports_early_init = False
    _layer_type = None

    def __init__(self):
        self.keys = None
        self.values = None

    def update(self, key_states, value_states, cache_kwargs=None):
        if self.keys is None:
            self.keys, self.values = key_states, value_states
            return self.keys, self.values
        cache_kwargs = cache_kwargs or {}
        head_lens = cache_kwargs["head_lens"]
        cu_klen = cache_kwargs["cu_klen"]
        dim = key_states.shape[-1]
        self.keys = _flat_append(self.keys.view(-1, dim), key_states.view(-1, dim),
                                 head_lens, cu_klen)
        self.values = _flat_append(self.values.view(-1, dim), value_states.view(-1, dim),
                                   head_lens, cu_klen)
        return self.keys, self.values

    # The flattened layout has no single sequence length. Report 1 so generation's
    # position bookkeeping advances by one token per step, matching the 4.44 path.
    def get_seq_length(self, *_a, **_k):
        return 0 if self.keys is None else 1

    def get_max_cache_shape(self, *_a, **_k):
        return None

    def get_max_length(self, *_a, **_k):
        return None

    def get_mask_sizes(self, query_length, *_a, **_k):
        """(kv_length, kv_offset) for mask construction -- matches DynamicLayer.

        The argument is a query LENGTH (an int), not a cache_position tensor; the
        earlier tensor-indexing version died with "'int' object is not subscriptable"
        the moment a mask was built. Returning the same shape as DynamicLayer is
        right even though this layer is ragged: at decode the hijack serves attention
        itself through flash_attn_varlen / the torch fallback, both of which take the
        real per-head extents from cu_klen, so this value only has to keep the
        model-level mask machinery consistent.
        """
        return self.get_seq_length() + int(query_length), 0

    def reset(self):
        self.keys = self.values = None

    def lazy_initialization(self, *_a, **_k):
        return None

    def offload(self, *_a, **_k):
        return None

    def prefetch(self, *_a, **_k):
        return None

    def crop(self, *_a, **_k):
        # Cropping means "keep the first N positions", which has no meaning once each
        # head holds a different, already-selected subset. Refuse rather than silently
        # corrupt the cache.
        raise NotImplementedError(
            "crop() is not defined for a ragged per-head cache (post-eviction)")

    def batch_repeat_interleave(self, *_a, **_k):
        raise NotImplementedError("batch ops are not defined for a ragged per-head cache")

    def batch_select_indices(self, *_a, **_k):
        raise NotImplementedError("batch ops are not defined for a ragged per-head cache")

    def reorder_cache(self, *_a, **_k):
        raise NotImplementedError("beam reordering is not defined for a ragged per-head cache")


# Register as a virtual subclass of CacheLayerMixin rather than inheriting it.
#
# transformers dispatches on `isinstance(layer, CacheLayerMixin)` -- notably in
# DynamicCache.get_seq_length, which scans for the first *attention* layer and
# raises "the current Cache seem to only contain LinearAttention layers" if it
# finds none. A duck-typed layer is invisible to that check, so on a hybrid model
# every compressed layer looked like a linear-attention layer and generation died.
#
# ABCMeta.register satisfies the isinstance check without inheriting DynamicLayer's
# behaviour -- which is the point: its get_seq_length reads `keys.shape[-2]` as a
# sequence length, and for a flattened ragged layout that number is meaningless.
CacheLayerMixin = None
try:
    from transformers.cache_utils import CacheLayerMixin as _CLM
    _CLM.register(_SplitHeadFlattenLayer)
    CacheLayerMixin = _CLM
except Exception:
    pass


def make_hybrid_model_forward(orig_forward):
    """Wrap a 5.x TextModel.forward so the final phase gets the ragged cache.

    The only thing the model-level hook must do is substitute the cache before the
    decoder stack runs: after compress(), the compressed layers no longer hold a
    rectangular [B,H,S,D] tensor, and the stock DynamicCache cannot represent that.
    Everything else (masks, rotary, layer loop) is left to the original forward --
    which is why this is a wrapper rather than a copied-out forward like the 4.44
    backbones needed. Less code to drift out of sync with transformers.
    """
    def forward(self, *args, **kwargs):
        if getattr(self.config, "lookahead_phase", "collect") == "final":
            pkv = kwargs.get("past_key_values")
            # Idempotent: generate() calls the model once per decoded token and
            # passes the cache back in. Rebuilding it every step would discard the
            # evicted cache and silently un-compress the run.
            if not getattr(pkv, "_lorekv_hybrid_flat", False):
                kwargs["past_key_values"] = make_hybrid_flat_cache(self.config, pkv)
        return orig_forward(self, *args, **kwargs)
    forward._lorekv_wrapped = True
    return forward


def _phase(self):
    return getattr(self.config, "lookahead_phase", "collect")


def _init_hybrid_snapkv(self):
    """init_lookahead_snapkv plus what the 5.x attention classes do not carry.

    The 4.44 *FlashAttention2 classes stored `self.num_heads`; Qwen3_5Attention and
    Gemma4UnifiedTextAttention do not (they derive shapes from -1 reshapes), so it
    is resolved from the config once. `base_capacity` is rewritten here to the
    per-compressible-layer budget, which is the ONE place the hybrid budget
    accounting is applied -- see hybrid_support.resolve_budget.
    """
    if not hasattr(self, "kv_cluster"):
        cfg = self.config
        text_cfg = getattr(cfg, "text_config", None) or cfg
        if not hasattr(cfg, "num_hidden_layers"):
            cfg.num_hidden_layers = int(text_cfg.num_hidden_layers)
        # Applied once, before the cluster is constructed, so every downstream use
        # (Eq. 19-20 allocation included) sees the resolved budget.
        if not getattr(cfg, "_lorekv_budget_resolved", False):
            cfg.base_capacity = resolve_budget(cfg, cfg.base_capacity)
            cfg._lorekv_budget_resolved = True
    init_lookahead_snapkv(self)
    if getattr(self.kv_cluster, "num_heads", None) is None:
        text_cfg = getattr(self.config, "text_config", None) or self.config
        self.kv_cluster.num_heads = int(text_cfg.num_attention_heads)
    # The ATTENTION MODULE's value is authoritative, not the config's. init_*_snapkv
    # computes num_key_value_groups as a global
    # `config.num_attention_heads // config.num_key_value_heads`, but Gemma 4 keeps
    # num_key_value_heads in per_layer_config, so the global read can disagree with
    # what this layer's projections actually emit. The cluster then repeat_kv's by the
    # wrong factor and the score matmul fails ("size of tensor a (4) must match tensor
    # b (2)"). self.num_key_value_groups was derived from this layer's own config when
    # the module was built, so it always matches the tensors.
    self.kv_cluster.num_key_value_groups = int(self.num_key_value_groups)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def _collect_or_lookahead(self, query_states, key_states, value_states, past_key_values):
    """Record what compress() will need, and return the FULL (uncompressed) K/V.

    Called for both the prefill ("collect") and the lookahead rollout. Eviction
    does not happen here: the lookahead continuations must be generated against the
    uncompressed cache before any token can be scored, so compress() runs later,
    driven by lookahead_runner once all M particles are done.
    """
    k_compact, v_compact = past_key_values.update(key_states, value_states, self.layer_idx)
    if _phase(self) == "collect":
        self.kv_cluster.capture_window_queries(query_states)
        # Store COMPACT (num_key_value_heads) K/V, not GQA-repeated: repeating here
        # would persist a tensor num_key_value_groups times larger on every layer for
        # the whole rollout. compress() repeats transiently, one layer at a time.
        self.kv_cluster.store_full_kv(k_compact, v_compact)
    else:
        self.kv_cluster.capture_lookahead_query(
            getattr(self.config, "lookahead_particle_idx", 0), query_states)
    return k_compact, v_compact


def _final_phase_varlen(self, query_states, key_states, value_states, past_key_values):
    """Decode against the ragged per-head cache left behind by eviction.

    After compress(), each head of a compressed layer keeps a DIFFERENT number of
    tokens (Eq. 19-20 allocate per-head budgets from perturbation mass). No dense
    [B,H,S,D] layout can express that, so the cache is stored flattened and served
    with the varlen kernel, exactly as on the 4.44 backbones.
    """
    cache_kwargs = {
        "head_lens": self.kv_cluster.head_lens,
        "cu_klen": self.kv_cluster.cu_klen,
    }
    n_rep = self.num_key_value_groups
    key_states = _repeat_kv(key_states, n_rep)
    value_states = _repeat_kv(value_states, n_rep)
    key_states, value_states = past_key_values.update(
        key_states, value_states, self.layer_idx, cache_kwargs)

    n_heads = self.kv_cluster.num_heads
    self.kv_cluster.klen_sum += n_heads
    self.kv_cluster.max_seqlen_k += 1
    self.kv_cluster.cu_klen += self.kv_cluster.cu_offset
    self.kv_cluster.head_lens += 1

    head_dim = self.head_dim
    q = query_states.view(-1, 1, head_dim)
    k = key_states.view(-1, 1, head_dim)
    v = value_states.view(-1, 1, head_dim)

    return _varlen_attend(q, k, v, self.kv_cluster.cu_qlen, self.kv_cluster.cu_klen,
                          self.kv_cluster.max_seqlen_k,
                          softmax_scale=getattr(self, "scaling", None))


# --------------------------------------------------------------------------
# Qwen3.5  (model_type "qwen3_5")
# --------------------------------------------------------------------------
def causal_lookahead_qwen3_5_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values=None,
    **kwargs,
):
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        apply_rotary_pos_emb, eager_attention_forward)
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    # Qwen3_5Attention only exists on full_attention layers, but assert the
    # invariant rather than trust it: a future checkpoint could reuse the class.
    if not layer_is_compressible(self.config, self.layer_idx):
        return self._lorekv_orig_forward(
            hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)

    _init_hybrid_snapkv(self)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # Qwen3.5 gates the attention output: q_proj emits 2*head_dim per head and the
    # second half becomes a sigmoid gate applied AFTER attention. Dropping the gate
    # would change the model, so it is carried through both phases untouched.
    query_states, gate = torch.chunk(
        self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)

    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if _phase(self) in ("collect", "lookahead"):
        k_compact, v_compact = _collect_or_lookahead(
            self, query_states, key_states, value_states, past_key_values)
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward)
        attn_output, _ = attention_interface(
            # NOT pre-repeated: the 5.x attention interface applies repeat_kv itself
            # (eager_attention_forward does it on entry), so repeating here would hand
            # it num_key_value_groups^2 heads.
            self, query_states, k_compact, v_compact,
            attention_mask, dropout=0.0, scaling=self.scaling, **kwargs)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    else:
        out = _final_phase_varlen(
            self, query_states, key_states, value_states, past_key_values)
        bsz, q_len = input_shape
        attn_output = out.reshape(bsz, self.kv_cluster.num_heads, q_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)

    attn_output = attn_output * torch.sigmoid(gate)
    return self.o_proj(attn_output), None


# --------------------------------------------------------------------------
# Gemma 4  (model_type "gemma4_unified")
# --------------------------------------------------------------------------
def causal_lookahead_gemma4_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    shared_kv_states: Optional[dict] = None,
    past_key_values=None,
    **kwargs,
):
    from transformers.models.gemma4_unified.modeling_gemma4_unified import (
        apply_rotary_pos_emb, eager_attention_forward)
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    # The decisive check for Gemma 4: one class serves sliding AND full layers.
    # Sliding layers are already O(1) in context (window=1024) -- compressing them
    # would be a different operation with different budget semantics.
    # is_kv_shared_layer also falls through: such a layer does not own its K/V, so
    # evicting there would corrupt the layer it borrows from.
    if not layer_is_compressible(self.config, self.layer_idx) or getattr(
            self, "is_kv_shared_layer", False):
        return self._lorekv_orig_forward(
            hidden_states, position_embeddings, attention_mask, shared_kv_states,
            past_key_values, **kwargs)

    _init_hybrid_snapkv(self)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    cos, sin = position_embeddings

    query_states = self.q_proj(hidden_states).view(hidden_shape)
    query_states = self.q_norm(query_states)
    query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2).transpose(1, 2)

    # attention_k_eq_v: on these layers v_proj is None and the value stream IS the
    # key stream (pre-norm). Reproduced exactly -- substituting a separate v_proj
    # would silently change the model.
    key_states = self.k_proj(hidden_states).view(hidden_shape)
    value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states
    key_states = self.k_norm(key_states)
    key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2).transpose(1, 2)
    value_states = self.v_norm(value_states).transpose(1, 2)

    if _phase(self) in ("collect", "lookahead"):
        k_compact, v_compact = _collect_or_lookahead(
            self, query_states, key_states, value_states, past_key_values)
        if getattr(self, "store_full_length_kv", False) and shared_kv_states is not None:
            shared_kv_states[self.layer_type] = (k_compact, v_compact)
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward)
        attn_output, _ = attention_interface(
            # NOT pre-repeated: the 5.x attention interface applies repeat_kv itself
            # (eager_attention_forward does it on entry), so repeating here would hand
            # it num_key_value_groups^2 heads.
            self, query_states, k_compact, v_compact,
            attention_mask, dropout=0.0, scaling=self.scaling,
            # Supplied by the stock Gemma 4 forward; None on the full-attention
            # layers we own, but omitting it silently changes behaviour if a
            # future checkpoint gives a compressed layer a window.
            sliding_window=getattr(self, "sliding_window", None), **kwargs)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    else:
        out = _final_phase_varlen(
            self, query_states, key_states, value_states, past_key_values)
        bsz, q_len = input_shape
        attn_output = out.reshape(bsz, self.kv_cluster.num_heads, q_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)

    return self.o_proj(attn_output), None
