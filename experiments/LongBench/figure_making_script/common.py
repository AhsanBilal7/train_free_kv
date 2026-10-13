"""🔬 Shared backbone for the LORE-KV mechanism-analysis figures (paper App. F).

Everything the four figure scripts need that touches the model lives here so the
compute scripts stay short and every figure is produced from the *same* captured
quantities (same prompts, same particles, same weights).

What this module gives you
--------------------------
* `parse_conf(path)`               - read a repo `.conf` file into a dict (first
                                     value of every space-separated sweep field).
* `add_repo_root()`                - put the repo root on sys.path so the proven
                                     `lorekv` package imports.
* `load_model_and_tokenizer(path)` - lorekv.harness, as in experiments/*/pred.py.
* `apply_lookahead_patch()`        - install LORE-KV's phase-aware forwards.
* `configure_model(model, conf)`   - push the `.conf` hyper-parameters onto the
                                     model config, exactly as pred.py does.
* `load_longbench_samples(...)`    - the same prompt-building path pred.py uses.
* `PostRoPEQueryCapture`           - non-invasive forward-pre-hooks that recompute
                                     the post-RoPE query states for every layer
                                     (used to record the TRUE full-cache decode
                                     queries Q_future without disturbing the graph).
* `collect_and_lookahead(...)`     - a faithful copy of phases 1-2 of
                                     `lookahead_runner.generate_with_lookahead`
                                     that STOPS before compress(), so the figures
                                     can read the captured window queries, the M
                                     lookahead particle queries, the full K/V, the
                                     per-particle (log-prob, entropy, trajectory
                                     divergence, weight), and the true decode queries.
* scoring helpers reused straight from `TestTimeLookaheadKVCluster`.

These load a 7B/8B model and need one GPU.
"""
import gc
import json
import math
import glob
import os
import pickle
import re
import sys
import tempfile

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Paths. This file lives at experiments/LongBench/figure_making_script/common.py
# so the experiment dir is two levels up and the repo root three levels up.
# ---------------------------------------------------------------------------
FIG_DIR = os.path.dirname(os.path.abspath(__file__))
LONGBENCH_DIR = os.path.dirname(FIG_DIR)              # experiments/LongBench
EXPERIMENTS_DIR = os.path.dirname(LONGBENCH_DIR)      # experiments
REPO_ROOT = os.path.dirname(EXPERIMENTS_DIR)          # repo root
CONFIG_DIR = os.path.join(REPO_ROOT, "config")


def add_repo_root():
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)


# ---------------------------------------------------------------------------
# .conf parsing (the format scripts/sweep.py reads: KEY = VALUE, # comments,
# blank lines ignored). Sweep fields hold several space-separated values; for a
# single-setting figure we take the first one, with an optional override map.
# ---------------------------------------------------------------------------
def parse_conf(path, overrides=None):
    conf = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            # strip trailing inline comments and take the first sweep value
            value = value.split("#", 1)[0].strip()
            if not value:
                continue
            conf[key] = value.split()[0]
    if overrides:
        conf.update({k: str(v) for k, v in overrides.items() if v is not None})
    return conf


def conf_get(conf, key, cast, default):
    if key in conf:
        return cast(conf[key])
    return default


# ---------------------------------------------------------------------------
# Model / patch / config: the same lorekv.harness calls experiments/*/pred.py uses.
# ---------------------------------------------------------------------------
def apply_lookahead_patch():
    add_repo_root()
    from lorekv.harness import install_method
    install_method("causal_lookahead")


def load_model_and_tokenizer(path):
    add_repo_root()
    from lorekv.harness import load_model_and_tokenizer as _load
    return _load(path)


def configure_model(model, conf, budget):
    """Write a parsed .conf's LORE-KV hyper-parameters onto the model, as pred.py does."""
    import argparse
    add_repo_root()
    from lorekv.harness import add_method_args, configure_model as _configure
    args = add_method_args(argparse.ArgumentParser()).parse_args([])
    args.budget = int(budget)
    for key, attr, cast in (("CHUNK_SIZE", "chunk_size", int),
                            ("NUM_PARTICLES", "num_particles", int),
                            ("LOOKAHEAD_LEN", "lookahead_len", int),
                            ("LOOKAHEAD_TEMPERATURE", "lookahead_temperature", float),
                            ("BETA", "beta", float), ("GAMMA", "gamma", float),
                            ("LAMBDA_D", "lambda_d", float),
                            ("NORMALIZE_TYPICALITY", "normalize_typicality", int),
                            ("ROBUST_ALPHA", "robust_alpha", float),
                            ("ROBUST_TOPK_BETA", "robust_topk_beta", float),
                            ("LAMBDA_LOOKAHEAD", "lambda_lookahead", float),
                            ("LAMBDA_K", "lambda_k", float), ("LAMBDA_V", "lambda_v", float),
                            ("BUDGET_TAU", "budget_tau", float)):
        if key in conf:
            setattr(args, attr, cast(conf[key]))
    _configure(model, args)
    model.model.config.divergence_layer_stride = conf_get(conf, "DIVERGENCE_LAYER_STRIDE", int, 1)
    return model


# ---------------------------------------------------------------------------
# Dataset. Single-doc QA is the default (paper's Section 4.2 grouping); --dataset
# full uses all 16 English LongBench tasks. Prompt building copies pred.get_pred.
# ---------------------------------------------------------------------------
SINGLE_DOC_QA = ["qasper", "narrativeqa", "multifieldqa_en"]
FULL_LONGBENCH = [
    "qasper", "narrativeqa", "multifieldqa_en",
    "hotpotqa", "2wikimqa", "musique",
    "trec", "triviaqa", "samsum",
    "gov_report", "qmsum", "multi_news",
    "passage_count", "passage_retrieval_en",
    "lcc", "repobench-p",
]


def resolve_datasets(name):
    if name in ("single", "single_qa", "singleqa"):
        return SINGLE_DOC_QA
    if name in ("full", "all", "longbench"):
        return FULL_LONGBENCH
    return name.split(",")


def _longbench_jsonl(dataset, dataset_path=None):
    add_repo_root()
    from lorekv.data import longbench_jsonl
    return longbench_jsonl(dataset, dataset_path)


def _build_chat(tokenizer, prompt, model_name):
    if "llama-3" in model_name.lower() and "instruct" in model_name.lower():
        msg = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    # Mistral-Instruct and the generic fallback
    return f"[INST]{prompt}[/INST]" if "mistral" in model_name.lower() else prompt


def load_samples(benchmark, tokenizer, dataset, model_name, dataset_path=None,
                 max_length=64000, samples_per_dataset=2, device="cuda:0"):
    """Benchmark-aware dispatcher used by every figure's compute script.

    benchmark == 'longbench' -> LongBench sub-tasks (default 'single' = single-doc QA).
    benchmark == 'ruler'     -> RULER sequence-length configs (default '8192')."""
    if benchmark == "ruler":
        path = dataset_path or "simonjegou/ruler"
        configs = resolve_ruler_configs(dataset)
        yield from load_ruler_samples(tokenizer, configs, model_name, dataset_path=path,
                                      max_length=max_length, samples_per_config=samples_per_dataset,
                                      device=device)
    else:
        path = dataset_path or "THUDM/LongBench"
        datasets = resolve_datasets(dataset)
        yield from load_longbench_samples(tokenizer, datasets, model_name, dataset_path=path,
                                          max_length=max_length, samples_per_dataset=samples_per_dataset,
                                          device=device)


def load_longbench_samples(tokenizer, datasets, model_name, dataset_path="THUDM/LongBench",
                           max_length=64000, samples_per_dataset=2, device="cuda:0"):
    """Yield dicts {dataset, max_gen, input_ids, attention_mask, context_length} using the
    identical prompt pipeline as experiments/LongBench/pred.py (truncate-in-the-middle,
    build_chat, per-dataset max_gen)."""
    from datasets import load_dataset
    d2prompt = json.load(open(os.path.join(LONGBENCH_DIR, "config", "dataset2prompt.json")))
    d2maxlen = json.load(open(os.path.join(LONGBENCH_DIR, "config", "dataset2maxlen.json")))
    no_chat = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}

    for dataset in datasets:
        jsonl = _longbench_jsonl(dataset, dataset_path)
        if jsonl is not None:
            # Offline path: same rows, same fields, no Hub round-trip.
            data = load_dataset("json", data_files=jsonl, split="train")
        else:
            # Match experiments/LongBench/pred.py's load call exactly, including data_dir --
            # that data_dir is what produced the hashed offline-cache config names
            # (e.g. 'qasper-ae14b6d15afac659'); omitting it misses the cache offline.
            data = load_dataset(dataset_path, dataset, split="test", data_dir=f"{dataset_path}/data")
        prompt_format = d2prompt[dataset]
        max_gen = int(d2maxlen[dataset])
        for i, obj in enumerate(data):
            if samples_per_dataset >= 0 and i >= samples_per_dataset:
                break                                     # negative -> run all samples
            prompt = prompt_format.format(**obj)
            ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if len(ids) > max_length:
                half = max_length // 2
                prompt = (tokenizer.decode(ids[:half], skip_special_tokens=True)
                          + tokenizer.decode(ids[-half:], skip_special_tokens=True))
            if dataset not in no_chat:
                prompt = _build_chat(tokenizer, prompt, model_name)
            enc = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            yield {
                "dataset": dataset,
                "sample_idx": i,
                "max_gen": max_gen,
                "input_ids": enc.input_ids,
                "attention_mask": enc.attention_mask,
                "context_length": enc.input_ids.shape[-1],
                "answers": obj.get("answers", []),
            }


# ---------------------------------------------------------------------------
# RULER samples. Same eviction machinery, RULER's own row format (context /
# question / answer_prefix), mirroring experiments/RULER/pred.build_ruler_input.
# ---------------------------------------------------------------------------
def resolve_ruler_configs(name):
    if name in ("single", "single_qa", "singleqa"):
        return ["8192"]
    if name in ("full", "all", "ruler"):
        return ["4096", "8192", "16384"]
    return name.split(",")


def load_ruler_samples(tokenizer, configs, model_name, dataset_path="simonjegou/ruler",
                       max_length=64000, samples_per_config=2, device="cuda:0"):
    from datasets import load_dataset
    for length in configs:
        data = load_dataset(dataset_path, str(length), split="test")
        for i, obj in enumerate(data):
            if samples_per_config >= 0 and i >= samples_per_config:
                break                                     # negative -> run all samples
            user_content = obj["context"] + obj["question"]
            if getattr(tokenizer, "chat_template", None):
                input_ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_content}],
                    add_generation_prompt=True, return_tensors="pt")
                if obj.get("answer_prefix"):
                    prefix = tokenizer(obj["answer_prefix"], add_special_tokens=False,
                                       return_tensors="pt").input_ids
                    input_ids = torch.cat([input_ids, prefix], dim=1)
            else:
                input_ids = tokenizer(user_content + obj.get("answer_prefix", ""),
                                      truncation=False, return_tensors="pt").input_ids
            if input_ids.shape[1] > max_length:
                half = max_length // 2
                input_ids = torch.cat([input_ids[:, :half], input_ids[:, -half:]], dim=1)
            input_ids = input_ids.to(device)
            ans = obj.get("answer", [])
            yield {
                "dataset": f"ruler-{length}",
                "sample_idx": i,
                "max_gen": int(obj.get("max_new_tokens", 32)),
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "context_length": input_ids.shape[-1],
                "answers": ans if isinstance(ans, list) else [ans],
            }


# ---------------------------------------------------------------------------
# Non-invasive post-RoPE query capture. Registers a forward-pre-hook on every
# attention module that recomputes q (post-RoPE) from the module's own weights
# and position_ids. Nothing about the forward pass changes; we only read inputs.
# ---------------------------------------------------------------------------
class PostRoPEQueryCapture:
    def __init__(self, model):
        add_repo_root()
        from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb
        self._apply_rope = apply_rotary_pos_emb
        self.model = model
        self.layers = model.model.layers
        self.enabled = False
        self.buffers = [[] for _ in self.layers]   # per layer: list of [1, Hq, q_len, D]
        self._handles = []

    def _make_hook(self, layer_idx):
        def hook(module, args, kwargs):
            if not self.enabled:
                return
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            position_ids = kwargs.get("position_ids", None)
            if hidden is None or position_ids is None:
                return
            bsz, q_len, _ = hidden.shape
            D = module.head_dim
            q = module.q_proj(hidden).view(bsz, q_len, module.num_heads, D).transpose(1, 2)
            k = module.k_proj(hidden).view(bsz, q_len, module.num_key_value_heads, D).transpose(1, 2)
            v = module.v_proj(hidden).view(bsz, q_len, module.num_key_value_heads, D).transpose(1, 2)
            cos, sin = module.rotary_emb(v, position_ids)
            q_rot, _ = self._apply_rope(q, k, cos, sin)
            self.buffers[layer_idx].append(q_rot.detach())
        return hook

    def __enter__(self):
        for li, layer in enumerate(self.layers):
            h = layer.self_attn.register_forward_pre_hook(self._make_hook(li), with_kwargs=True)
            self._handles.append(h)
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []

    def reset(self):
        self.buffers = [[] for _ in self.layers]

    def stacked(self):
        """Return per-layer [1, Hq, Tsteps, D] concatenation of captured queries."""
        out = []
        for buf in self.buffers:
            out.append(torch.cat(buf, dim=2) if buf else None)
        return out


# ---------------------------------------------------------------------------
# Phases 1-2 of lookahead_runner.generate_with_lookahead, stopped before compress().
# ---------------------------------------------------------------------------
def _clone_dynamic_cache(cache):
    from transformers.cache_utils import DynamicCache
    new = DynamicCache()
    for li in range(len(cache.key_cache)):
        new.key_cache.append(cache.key_cache[li].clone())
        new.value_cache.append(cache.value_cache[li].clone())
    new._seen_tokens = cache._seen_tokens
    return new


@torch.no_grad()
def collect_and_lookahead(model, input_ids, attention_mask, decode_steps=32, capture_true=True):
    """OOM-safe wrapper: if a sample is too large for the GPU, release everything and return
    a skip sentinel (true_queries=None, per_layer=[]) so the caller drops it and continues,
    instead of crashing the whole run. Otherwise returns the full capture dict."""
    try:
        return _collect_and_lookahead_impl(model, input_ids, attention_mask, decode_steps, capture_true)
    except torch.cuda.OutOfMemoryError:
        cfg = model.model.config
        M = cfg.num_particles
        for layer in model.model.layers:
            kvc = layer.self_attn.kv_cluster
            kvc.reset_lookahead_buffers()
            kvc.release_collected_state()
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[oom] sample too large for GPU (n={int(input_ids.shape[1])}); skipping and continuing. "
              f"Lower --max-length or set FIG_MAX_ANALYSIS_TOKENS to include long samples.", flush=True)
        T = cfg.lookahead_len
        return {"n": int(input_ids.shape[1]), "M": M, "T": T,
                "particle_logprob": torch.zeros(M), "particle_entropy": torch.zeros(M),
                "particle_c": torch.zeros(M, T), "particle_H": torch.zeros(M, T),
                "particle_V": torch.zeros(M, T),
                "particle_c_bar": torch.zeros(M), "particle_V_bar": torch.zeros(M),
                "particle_divergence": torch.zeros(M), "particle_divergence_z": torch.zeros(M),
                "particle_weights": torch.zeros(M), "particle_R": torch.zeros(M),
                "beta": float(getattr(cfg, "particle_beta", 1.0)),
                "gamma": float(getattr(cfg, "particle_gamma", 0.0)),
                "lambda_d": float(getattr(cfg, "lambda_d", 0.0)),
                "normalize": bool(getattr(cfg, "normalize_typicality", False)),
                "true_queries": None, "per_layer": []}


@torch.no_grad()
def _collect_and_lookahead_impl(model, input_ids, attention_mask, decode_steps=32, capture_true=True):
    """Run collect prefill + M lookahead particles + (optionally) a greedy full-cache decode
    for the TRUE future queries. Returns a dict of captured, per-layer analysis tensors.

    Faithful to phases 1-2 of lookahead_runner.generate_with_lookahead; the only additions are
    (a) recording the PER-STEP statistics (c_u, H_u, V_u) and per-particle logprob/entropy
        separately, not just the combined R_m -- tests V3/V4/V5 need the sequence, and
        cbar_m = logp_mean + H_mean is exactly recoverable from the two means alone, and
    (b) the true-decode capture via PostRoPEQueryCapture.
    """
    device = input_ids.device
    # Optional analysis-length cap (env FIG_MAX_ANALYSIS_TOKENS): the mechanism figures
    # don't need the full 40K context, and prefill + M cache-clones for that length OOMs
    # a shared GPU. Middle-truncate (keep head+tail) so very long samples still contribute
    # at a memory-safe length. 0/unset = no cap (full length).
    _cap = int(os.environ.get("FIG_MAX_ANALYSIS_TOKENS", "0") or 0)
    if _cap and input_ids.shape[1] > _cap:
        print(f"[analysis] truncating prompt {input_ids.shape[1]} -> {_cap} tokens "
              f"(FIG_MAX_ANALYSIS_TOKENS)", flush=True)
        half = _cap // 2
        input_ids = torch.cat([input_ids[:, :half], input_ids[:, -half:]], dim=1)
        attention_mask = torch.ones_like(input_ids) if attention_mask is not None else None
    n = input_ids.shape[1]
    cfg = model.model.config
    M = cfg.num_particles
    T = cfg.lookahead_len
    beta = getattr(cfg, "particle_beta", 1.0)
    gamma = getattr(cfg, "particle_gamma", 0.0)
    lambda_d = getattr(cfg, "lambda_d", 0.0)
    normalize = bool(getattr(cfg, "normalize_typicality", False))
    div_stride = getattr(cfg, "divergence_layer_stride", 1)
    temperature = cfg.lookahead_temperature
    top_k = getattr(cfg, "lookahead_top_k", None)
    top_p = getattr(cfg, "lookahead_top_p", None)

    # ---- Phase 1: full uncompressed prefill (window queries + full K/V captured per layer) ----
    cfg.lookahead_phase = "collect"
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    base_cache = out.past_key_values
    last_logits = out.logits[:, -1, :].float()

    # ---- TRUE future queries: greedy full-cache decode, captured non-invasively ----
    true_queries = None
    if capture_true and decode_steps > 0:
        cap = PostRoPEQueryCapture(model)
        cfg.lookahead_phase = "lookahead"          # stock full-attention path, stock DynamicCache
        with cap:
            cap.enabled = True
            tcache = _clone_dynamic_cache(base_cache)
            tok = last_logits.argmax(dim=-1, keepdim=True)
            pos = n
            for _ in range(decode_steps):
                step = model(input_ids=tok, past_key_values=tcache, use_cache=True,
                             cache_position=torch.tensor([pos], device=device),
                             position_ids=torch.tensor([[pos]], device=device))
                tcache = step.past_key_values
                tok = step.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                pos += 1
            cap.enabled = False
        true_queries = cap.stacked()               # per layer [1, Hq, decode_steps, D]
        # capture_lookahead_query polluted particle buffer 0 during this pass; clear it.
        for layer in model.model.layers:
            layer.self_attn.kv_cluster.reset_lookahead_buffers()

    # ---- Phase 2: M sampled lookahead particles (queries captured onto each kv_cluster) ----
    cfg.lookahead_phase = "lookahead"
    add_repo_root()
    from lorekv.monkeypatch.lookahead_runner import _shape_logits
    from lorekv.monkeypatch.scoring import step_stats
    particle_lp = torch.zeros(M, T, device=device)   # log q(y_u | h_u)
    particle_c = torch.zeros(M, T, device=device)    # centered score c_u = log q + H
    particle_H = torch.zeros(M, T, device=device)    # entropy H_u
    particle_V = torch.zeros(M, T, device=device)    # varentropy V_u
    for m in range(M):
        cfg.lookahead_particle_idx = m
        cache_m = _clone_dynamic_cache(base_cache)
        q0 = torch.log_softmax(_shape_logits(last_logits, temperature, top_k, top_p), dim=-1).exp()
        tok = torch.multinomial(q0, num_samples=1)
        lps, cs, hs, vs = [], [], [], []
        pos = n
        for _t in range(T):
            step = model(input_ids=tok, past_key_values=cache_m, use_cache=True,
                         cache_position=torch.tensor([pos], device=device),
                         position_ids=torch.tensor([[pos]], device=device))
            logits = step.logits[:, -1, :].float()
            z = _shape_logits(logits, temperature, top_k, top_p)
            next_tok = torch.multinomial(torch.softmax(z, dim=-1), num_samples=1)
            # Centering law == sampling law, so E[c_u | h_u] = 0 holds exactly.
            lp_u, h_u, v_u, c_u = step_stats(logits, next_tok.squeeze(-1),
                                             temperature=temperature, top_k=top_k, top_p=top_p)
            lps.append(lp_u.squeeze()); cs.append(c_u.squeeze())
            hs.append(h_u.squeeze()); vs.append(v_u.squeeze())
            cache_m = step.past_key_values
            tok = next_tok
            pos += 1
        particle_lp[m] = torch.stack(lps)
        particle_c[m] = torch.stack(cs)
        particle_H[m] = torch.stack(hs)
        particle_V[m] = torch.stack(vs)
    particle_logprob = particle_lp.mean(-1)          # kept: the figures index it by name
    particle_entropy = particle_H.mean(-1)

    # Eq. 10 / App. C.3: trajectory disagreement D_m, computed from the SAME per-layer lookahead
    # query buffers the runner uses (so the figures and the deployed method score
    # identically). Always computed here -- even when lambda_D == 0 -- because test S4
    # (the incremental value of disagreement over (cbar, Hbar)) must run on ANY capture.
    from lorekv.monkeypatch.scoring import (
        particle_trajectory_divergence, reliability_weights, standardize_disagreement,
    )
    particle_divergence = particle_trajectory_divergence(
        model.model.layers, M, layer_stride=div_stride, device=device)
    particle_divergence_z = standardize_disagreement(particle_divergence)

    # Eqs. 3-4: reliability score and normalized weights (matches the runner / paper).
    weights, parts = reliability_weights(
        particle_c, particle_H, particle_V, particle_divergence_z,
        beta=beta, gamma=gamma, lambda_d=lambda_d, normalize=normalize)
    for layer in model.model.layers:
        layer.self_attn.kv_cluster.set_particle_weights(weights)

    # Snapshot the per-layer captured tensors (queries live on the kv_cluster objects).
    per_layer = []
    for layer in model.model.layers:
        kvc = layer.self_attn.kv_cluster
        look = []
        for m in range(M):
            buf = kvc._lookahead_buffers[m]
            look.append(torch.cat(buf, dim=2) if buf else None)
        per_layer.append({
            "window_queries": kvc.window_queries,          # [1, Hq, w, D]
            "lookahead_queries": look,                     # list M of [1, Hq, T, D]
            "full_key_states": kvc.full_key_states,        # [1, Hkv, n, D] compact
            "full_value_states": kvc.full_value_states,    # [1, Hkv, n, D] compact
            "num_key_value_groups": kvc.num_key_value_groups,
            "o_proj_weight": kvc.o_proj_weight,
            "head_dim": kvc.head_dim,
        })

    del base_cache
    torch.cuda.empty_cache()
    return {
        "n": n,
        "M": M, "T": T,
        "particle_logprob": particle_logprob.cpu(),      # mean_u log q(y_u|h_u)
        "particle_entropy": particle_entropy.cpu(),      # Hbar_m
        "particle_logp_steps": particle_lp.cpu(),        # [M, T] per-step log q
        "particle_c": particle_c.cpu(),                  # [M, T] centered score c_u
        "particle_H": particle_H.cpu(),                  # [M, T] entropy H_u
        "particle_V": particle_V.cpu(),                  # [M, T] varentropy V_u
        "particle_c_bar": parts["c_bar"].cpu(),          # cbar_m == logp_mean + H_mean
        "particle_V_bar": parts["V_bar"].cpu(),
        "particle_typicality": parts["typicality"].cpu(),
        "particle_R": parts["R"].cpu(),
        "particle_divergence": particle_divergence.cpu(),
        "particle_divergence_z": particle_divergence_z.cpu(),
        "particle_weights": weights.cpu(),
        "beta": float(beta), "gamma": float(gamma), "lambda_d": float(lambda_d),
        "normalize": bool(normalize),
        "true_queries": true_queries,     # per layer [1, Hq, decode_steps, D] or None
        "per_layer": per_layer,
    }


def cleanup_clusters(model):
    """Drop the big per-layer full K/V and query buffers between samples (matches
    kv_cluster.release_collected_state + reset_lookahead_buffers in the runner)."""
    for layer in model.model.layers:
        kvc = layer.self_attn.kv_cluster
        kvc.reset_lookahead_buffers()
        kvc.release_collected_state()
    torch.cuda.empty_cache()


def free_sample(model, cap=None):
    """Fully release a finished sample's GPU memory BEFORE the next one is processed.

    cleanup_clusters() only clears the per-layer kv_cluster tensors, but the dict
    returned by collect_and_lookahead still holds the true-query tensors AND per-layer
    references to those same K/V tensors -- so nothing is actually freed until `cap`
    is dropped. This clears `cap` too, then forces a GC pass and empties the CUDA
    allocator cache. Without it, a finished (up to ~40K-token) sample's memory carries
    into the next sample's allocation and OOMs when a longer dataset (e.g. NarrativeQA)
    follows a shorter one (e.g. Qasper)."""
    for layer in model.model.layers:
        kvc = layer.self_attn.kv_cluster
        kvc.reset_lookahead_buffers()
        kvc.release_collected_state()
    if isinstance(cap, dict):
        cap.clear()
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Cosine-alignment metric (Figure 1 / Figure 4): weighted coverage of the TRUE
# decode queries by an estimated query set.
#
#   coverage = mean_head  mean_t  sum_s w_s * max_{q in source_s} cos(q_true_t, q)
#
# A single source (weight 1) recovers the plain max-cosine coverage the proposal
# describes; weighting sources lets a confident-but-wrong particle be down-weighted.
# ---------------------------------------------------------------------------
def _unit(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


@torch.no_grad()
def weighted_coverage(true_q, sources, weights):
    """true_q: [Hq, Tt, D]; sources: list of [Hq, Qi, D]; weights: 1D len == len(sources).
    Returns a scalar float (mean over heads and true-query steps)."""
    Hq = true_q.shape[0]
    tq = _unit(true_q.float())                       # [Hq, Tt, D]
    per_source = []
    for src in sources:
        if src is None or src.shape[1] == 0:
            per_source.append(torch.zeros(Hq, tq.shape[1], device=tq.device))
            continue
        s = _unit(src.float())                       # [Hq, Qi, D]
        sim = torch.matmul(tq, s.transpose(1, 2))    # [Hq, Tt, Qi]
        per_source.append(sim.max(dim=-1).values)    # [Hq, Tt]
    stack = torch.stack(per_source, dim=0)           # [S, Hq, Tt]
    w = weights.to(stack.device).view(-1, 1, 1)
    cov = (w * stack).sum(dim=0)                      # [Hq, Tt]
    return float(cov.mean().item())


def softmax_weights(scores, eta=1.0):
    """Kept for back-compatibility. eta is NOT identifiable alongside (beta, gamma,
    lambda_D) -- softmax(eta*(b*c - g*H - lD*D)) == softmax((eta*b)*c - ...) -- so new
    code should call reliability_weights_from_capture() and leave eta at 1."""
    return F.softmax(eta * scores, dim=0)


# ---------------------------------------------------------------------------
# Trajectory reliability (Eqs. 2-3), evaluated on a capture.
#
#   c_u  = log q(y_u | h_u) + H(q(. | h_u)),  E[c_u | h_u] = 0
#   R_m  = beta * cbar_m - gamma * Hbar_m - lambda_D * Dhat_m
#
# cbar_m = (mean log-prob)_m + (mean entropy)_m is recoverable from any capture, so
# every score-level sweep below re-weights the same captured trajectories without
# re-decoding. Settings in (beta, gamma, lambda_D) coordinates (App. C.3, Table 4):
#
#   uniform weights                     (0,   0,       0)
#   log-probability only                (1,   1,       0)
#   LORE-KV /E                          (1,   1.3,     0)
#   LORE-KV /D                          (1,   1,       0.3)
#   typicality only                     (1,   0,       0)
#   disagreement without entropy        (1,   0,       0.3)
# ---------------------------------------------------------------------------
def centered_score(logprob_mean, entropy_mean):
    """cbar_m = mean_u log q(y_u|h_u) + mean_u H_u."""
    return logprob_mean + entropy_mean


def reliability_R(c_bar, H_bar, d_hat=None, beta=1.0, gamma=0.0, lambda_d=0.0,
                  V_bar=None, T=None, normalize=False):
    """R_m = beta * (cbar or ctilde) - gamma * Hbar - lambda_D * Dhat."""
    x = c_bar
    if normalize:
        if V_bar is None or not T:
            raise ValueError("normalize=True needs V_bar and T")
        x = c_bar / (V_bar / float(T)).clamp_min(1e-8).sqrt()
    R = beta * x - gamma * H_bar
    if d_hat is not None and lambda_d:
        R = R - lambda_d * d_hat
    return R


def reliability_weights_from_capture(c_bar, H_bar, d_hat=None, **kw):
    return F.softmax(reliability_R(c_bar, H_bar, d_hat, **kw), dim=0)


# The (beta, gamma, lambda_D) grid every score-level figure compares; gamma_E is
# LORE-KV /E's entropy coefficient.
def weighting_grid(gamma_E=1.3, lambda_D=0.3):
    return {
        "uniform":     (0.0, 0.0,     0.0),
        "logprob":     (1.0, 1.0,     0.0),
        "typicality":  (1.0, 0.0,     0.0),
        "entropy":     (1.0, gamma_E, 0.0),
        "divergence":  (1.0, 1.0,     lambda_D),
        "typ_div":     (1.0, 0.0,     lambda_D),
    }


# ---------------------------------------------------------------------------
# Trajectory disagreement D_m (Eq. 10) recomputed from an already-captured sample.
#
# collect_and_lookahead() returns D over the FULL ensemble it generated; figures that
# sweep nested prefixes (fig1's M in {1,2,4,8}) need D restricted to the prefix, since
# "how atypical is m" is defined relative to whichever particles are actually being
# combined. This runs the same core math on the captured per-layer query tensors.
# ---------------------------------------------------------------------------
@torch.no_grad()
def divergence_from_capture(per_layer, num_particles, layer_stride=1, device=None):
    """per_layer: the capture's list of per-layer dicts. Returns D [M] (zeros if M < 2)."""
    add_repo_root()
    from lorekv.monkeypatch.scoring import divergence_from_query_stacks

    M = int(num_particles)

    def _stacks():
        for pl in per_layer[::max(1, int(layer_stride))]:
            qs = pl.get("lookahead_queries")
            if not qs or len(qs) < M or any(qs[m] is None for m in range(M)):
                continue
            T = min(int(qs[m].shape[2]) for m in range(M))
            if T == 0:
                continue
            yield torch.stack([qs[m][0, :, :T, :] for m in range(M)], dim=0)

    return divergence_from_query_stacks(_stacks(), M, device=device)


def standardize_scores(x):
    """Zero-mean/unit-std across particles, exactly as R_m consumes D_m in the runner."""
    add_repo_root()
    from lorekv.monkeypatch.scoring import standardize_particle_scores
    return standardize_particle_scores(x)


# ---------------------------------------------------------------------------
# Scoring convenience helpers (shared by figs 2,3,5,6,8,9). They wrap the proven
# per-layer TestTimeLookaheadKVCluster scoring so every figure scores identically.
# ---------------------------------------------------------------------------
def get_repeated_kv(kvc):
    """GQA-repeat a layer's compact full K/V to query-head count -> ([1,H,n,D], [1,H,n,D])."""
    add_repo_root()
    from lorekv.monkeypatch.scoring import repeat_kv
    key = repeat_kv(kvc.full_key_states, kvc.num_key_value_groups)
    val = repeat_kv(kvc.full_value_states, kvc.num_key_value_groups)
    return key, val


def make_lookahead_batches(kvc, M):
    """Return (query_batches, causal_flags) = [window] + [particle_0..M-1], matching compress()."""
    batches = [kvc.window_queries]
    flags = [True]
    for m in range(M):
        buf = kvc._lookahead_buffers[m]
        q_m = torch.cat(buf, dim=2) if buf else kvc.window_queries[:, :, :0, :]
        batches.append(q_m)
        flags.append(False)
    return batches, flags


def combine_lookahead_scores(scores, weights, lambda_L, M):
    """Given _project_and_score output [window, p0..pM-1], build the window-only, single-
    trajectory, and confidence-weighted multi scores (Eq. 7). Each is [H, n]."""
    s_window = scores[0]
    s_single = s_window + lambda_L * scores[1] if M >= 1 else s_window.clone()
    s_multi = s_window.clone()
    for m in range(M):
        s_multi = s_multi + lambda_L * float(weights[m]) * scores[1 + m]
    return s_window, s_single, s_multi


@torch.no_grad()
def oracle_scores(kvc, key, val, true_queries):
    """Perturbation score under the TRUE full-cache decode queries -> the oracle
    importance S* every proxy is trying to match. true_queries: [1,Hq,Tt,D]. -> [H,n]."""
    return kvc._project_and_score(key, val, [true_queries], [False])[0]


@torch.no_grad()
def head_topb_mask(scores, n, budget, window_size):
    """scores:[H,n] -> bool mask [H,n] keeping token 0, the last `window_size` tokens,
    and the top (budget) candidates per head (protected positions forced in)."""
    H = scores.shape[0]
    device = scores.device
    protected = torch.zeros(H, n, dtype=torch.bool, device=device)
    protected[:, 0] = True
    protected[:, max(0, n - window_size):] = True
    s = scores.clone()
    s[protected] = float("inf")
    b = min(budget, n)
    idx = s.topk(b, dim=-1).indices
    mask = torch.zeros(H, n, dtype=torch.bool, device=device)
    mask.scatter_(1, idx, True)
    return mask


@torch.no_grad()
def attn_outputs(q, k, v, mask=None):
    """q:[H,Tt,D] k,v:[H,n,D] mask:[H,n] bool or None -> attention output o:[H,Tt,D]."""
    D = q.shape[-1]
    logits = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(D)   # [H,Tt,n]
    if mask is not None:
        logits = logits.masked_fill(~mask.unsqueeze(1), float("-inf"))
    a = F.softmax(logits, dim=-1)
    return torch.matmul(a, v)


def recall_at_b(pred_score, oracle_score, budget):
    """Fraction of the oracle's top-B tokens that the predicted score also ranks in its
    top-B (1-D numpy arrays, same length). Symmetric to top-B overlap when both are B."""
    import numpy as _np
    B = min(int(budget), len(pred_score))
    top_pred = set(_np.argsort(-pred_score)[:B].tolist())
    top_oracle = set(_np.argsort(-oracle_score)[:B].tolist())
    return len(top_pred & top_oracle) / max(1, B)


# ---------------------------------------------------------------------------
# Self-describing filenames. Every figure's .npz and .png embed the config name,
# model name, budget, and dataset label so sweeps don't overwrite each other and
# each artifact says exactly how it was produced. Layout:
#     <data_base>__<conf>__<model>__b<budget>__<dataset>.npz
#     <fig_base>__<conf>__<model>__b<budget>__<dataset>.png   (same tag)
# ---------------------------------------------------------------------------
def _sanitize(s):
    return re.sub(r"[^A-Za-z0-9.=-]+", "-", str(s)).strip("-")


def build_tag(conf_path, model_path, budget, dataset):
    """A filesystem-safe tag string identifying config / model / budget / dataset."""
    conf = os.path.splitext(os.path.basename(str(conf_path)))[0] if conf_path else "conf"
    model = str(model_path).rstrip("/").split("/")[-1]
    b = str(budget)
    if not b.startswith("b"):
        b = "b" + b
    parts = [_sanitize(conf), _sanitize(model), _sanitize(b), _sanitize(dataset)]
    return "__".join(parts)


def default_npz_path(out, out_dir, here, data_base, tag):
    """Explicit --out wins; else <out_dir or here>/<data_base>__<tag>.npz."""
    if out:
        return out
    d = out_dir or here
    return os.path.join(d, f"{data_base}__{tag}.npz")


def pick_npz(data, data_dir, here, data_base):
    """Explicit --data wins; else the newest <data_base>__*.npz in <data_dir or here>
    (falling back to the legacy untagged <data_base>.npz)."""
    if data:
        return data
    d = data_dir or here
    cands = sorted(glob.glob(os.path.join(d, f"{data_base}__*.npz")), key=os.path.getmtime)
    if cands:
        return cands[-1]
    return os.path.join(d, f"{data_base}.npz")


def png_path_for(npz_path, fig_base, out):
    """Explicit --out wins; else write the PNG next to the NPZ, carrying its tag:
    <dir(npz)>/<fig_base>__<tag>.png (tag = everything after the first '__')."""
    if out:
        return out
    d = os.path.dirname(os.path.abspath(npz_path))
    base = os.path.basename(npz_path)
    tag = base.split("__", 1)[1].rsplit(".npz", 1)[0] if "__" in base else ""
    return os.path.join(d, f"{fig_base}__{tag}.png" if tag else f"{fig_base}.png")


# ---------------------------------------------------------------------------
# Resumable per-entry checkpointing. The heavy figure computes run per sample;
# each one keeps its running accumulators in a plain dict and, after finishing a
# sample, flushes that dict + the set of finished entry keys to a pickle on
# SCRATCH (atomically). On the next launch the compute reloads that pickle and
# skips already-finished entries, so an interrupted run resumes instead of
# restarting. The FINAL .npz still lands in the requested (main) folder; the
# scratch checkpoint is only the crash-safe running copy and is removed on
# success.
# ---------------------------------------------------------------------------
def scratch_ckpt_dir():
    """Where the per-entry running checkpoints live (scratch, never $HOME)."""
    d = os.environ.get("FIG_CKPT_DIR")
    if d:
        return d
    user = os.environ.get("USER", "user")
    if os.path.isdir(f"/scratch/{user}"):
        return f"/scratch/{user}/lorekv_fig_ckpt"
    base = os.environ.get("TMPDIR") or tempfile.gettempdir()
    return os.path.join(base, "lorekv_fig_ckpt")


class Checkpoint:
    """Crash-safe resume for a single compute run, keyed to its final .npz name.

    Usage:
        ckpt = common.Checkpoint(args.out)
        state, done = ckpt.load_or({... fresh accumulators ...})
        for sample in ...:
            key = common.entry_key(sample)
            if key in done: continue
            ... update state[...] ...
            done.add(key); ckpt.flush(state, done)
        ... build final arrays from state ...; np.savez(args.out, **save)
        ckpt.done()          # remove the scratch checkpoint
    """
    def __init__(self, final_out):
        d = scratch_ckpt_dir()
        os.makedirs(d, exist_ok=True)
        base = os.path.splitext(os.path.basename(str(final_out)))[0]
        # Hash the FULL destination path so two runs that share a basename but live in
        # different run folders (e.g. different budgets) never share a checkpoint.
        import hashlib
        h = hashlib.md5(os.path.abspath(str(final_out)).encode()).hexdigest()[:10]
        self.final_out = final_out
        self.path = os.path.join(d, f"{base}__{h}.ckpt.pkl")

    def load_or(self, fresh_state):
        """Return (state, done_set): the resumed accumulators + finished keys, or a
        fresh copy of `fresh_state` and an empty set if there is no checkpoint."""
        if os.path.exists(self.path):
            try:
                with open(self.path, "rb") as f:
                    blob = pickle.load(f)
                print(f"[ckpt] resuming from {self.path} ({len(blob['done'])} entries done)", flush=True)
                return blob["state"], set(blob["done"])
            except Exception as e:  # corrupt/partial pickle -> start clean
                print(f"[ckpt] could not read {self.path} ({e}); starting fresh", flush=True)
        return fresh_state, set()

    def flush(self, state, done):
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"state": state, "done": list(done)}, f)
        os.replace(tmp, self.path)          # atomic on POSIX

    def done(self):
        try:
            os.remove(self.path)
        except OSError:
            pass


def entry_key(sample):
    """Stable per-entry id used to mark an entry finished (dataset + sample index)."""
    return f"{sample['dataset']}#{sample['sample_idx']}"
