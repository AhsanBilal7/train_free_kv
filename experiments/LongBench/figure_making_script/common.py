"""Shared backbone for the LORE-KV / TLC-KV analysis figures.

Everything the four figure scripts need that touches the model lives here so the
compute scripts stay short and every figure is produced from the *same* captured
quantities (same prompts, same particles, same weights).

What this module gives you
--------------------------
* `parse_conf(path)`               - read a repo `.conf` file into a dict (first
                                     value of every space-separated sweep field).
* `add_repo_root()`                - put the repo root on sys.path so the proven
                                     `causal_lookahead_snapkv` package imports.
* `load_model_and_tokenizer(path)` - identical settings to experiments/*/pred.py.
* `apply_lookahead_patch()`        - monkeypatch Mistral/Llama with the paper's
                                     phase-aware forward (from the method code).
* `configure_model(model, conf)`   - push the `.conf` hyper-parameters onto
                                     `model.model.config`, exactly as pred.py does.
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
                                     per-particle (log-prob, entropy, weight), and
                                     the true decode queries.
* scoring helpers reused straight from `TestTimeLookaheadKVCluster`.

No GPU is available on the login node; run these on the cluster (venv kv44).
"""
import json
import math
import glob
import os
import re
import sys

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
# .conf parsing (same format scripts/run_h100.sh reads: KEY = VALUE, # comments,
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
# Model / patch / config, matching experiments/LongBench/pred.py exactly.
# ---------------------------------------------------------------------------
def apply_lookahead_patch():
    add_repo_root()
    from causal_lookahead_snapkv.monkeypatch.monkeypatch import (
        replace_mistral_lookahead,
        replace_llama_lookahead,
    )
    replace_mistral_lookahead()
    replace_llama_lookahead()


def load_model_and_tokenizer(path):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, device_map="cuda:0")
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    return model.eval(), tokenizer


def configure_model(model, conf, budget):
    """Push .conf hyper-parameters onto model.model.config (mirrors pred.config_compress)."""
    c = model.model.config
    c.window_size = 32
    c.base_capacity = int(budget)
    c.kernel_size = 7
    c.pooling = "maxpool"
    c.floor_alpha = 0.5
    c.skip = 0
    c.normalize = None
    c.pyram_mode = False
    c.pyram_beta = 20
    c.gqa_support = False
    c.gqa_func = "mean"
    c.chunk_size = conf_get(conf, "CHUNK_SIZE", int, 4)
    c.mode = "causal_lookahead"
    c.lambda_red = conf_get(conf, "LAMBDA_RED", float, 0.0)

    c.lookahead_phase = "collect"
    c.lookahead_particle_idx = 0
    c.num_particles = conf_get(conf, "NUM_PARTICLES", int, 4)
    c.lookahead_len = conf_get(conf, "LOOKAHEAD_LEN", int, 8)
    c.lookahead_temperature = conf_get(conf, "LOOKAHEAD_TEMPERATURE", float, 1.0)
    c.lambda_h = conf_get(conf, "LAMBDA_H", float, 0.0)
    c.particle_eta = conf_get(conf, "PARTICLE_ETA", float, 1.0)
    c.robust_alpha = conf_get(conf, "ROBUST_ALPHA", float, 0.5)
    c.robust_topk_beta = conf_get(conf, "ROBUST_TOPK_BETA", float, 0.5)
    c.lambda_lookahead = conf_get(conf, "LAMBDA_LOOKAHEAD", float, 1.0)
    c.lambda_k = conf_get(conf, "LAMBDA_K", float, 0.0)
    c.lambda_v = conf_get(conf, "LAMBDA_V", float, 0.0)
    c.budget_tau = conf_get(conf, "BUDGET_TAU", float, 1.0)
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
    """Run collect prefill + M lookahead particles + (optionally) a greedy full-cache decode
    for the TRUE future queries. Returns a dict of captured, per-layer analysis tensors.

    Faithful to phases 1-2 of lookahead_runner.generate_with_lookahead; the only additions are
    (a) recording per-particle logprob & entropy separately (not just the combined R_m), and
    (b) the true-decode capture via PostRoPEQueryCapture.
    """
    device = input_ids.device
    n = input_ids.shape[1]
    cfg = model.model.config
    M = cfg.num_particles
    T = cfg.lookahead_len
    lambda_h = cfg.lambda_h
    eta = cfg.particle_eta
    temperature = cfg.lookahead_temperature

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
    particle_logprob = torch.zeros(M, device=device)
    particle_entropy = torch.zeros(M, device=device)
    for m in range(M):
        cfg.lookahead_particle_idx = m
        cache_m = _clone_dynamic_cache(base_cache)
        probs0 = F.softmax(last_logits / max(temperature, 1e-5), dim=-1)
        tok = torch.multinomial(probs0, num_samples=1)
        logps, ents = [], []
        pos = n
        for _t in range(T):
            step = model(input_ids=tok, past_key_values=cache_m, use_cache=True,
                         cache_position=torch.tensor([pos], device=device),
                         position_ids=torch.tensor([[pos]], device=device))
            logits = step.logits[:, -1, :].float()
            logprobs = F.log_softmax(logits, dim=-1)
            probs = logprobs.exp()
            ent = -(probs * logprobs).sum(-1)
            next_tok = torch.multinomial(F.softmax(logits / max(temperature, 1e-5), dim=-1), num_samples=1)
            logps.append(logprobs.gather(-1, next_tok).squeeze())
            ents.append(ent.squeeze())
            cache_m = step.past_key_values
            tok = next_tok
            pos += 1
        particle_logprob[m] = torch.stack(logps).mean()
        particle_entropy[m] = torch.stack(ents).mean()

    # Eq. 5-6: combined confidence and normalized weights (matches the runner / paper).
    particle_scores = particle_logprob - lambda_h * particle_entropy
    weights = F.softmax(eta * particle_scores, dim=0)
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
        "particle_logprob": particle_logprob.cpu(),
        "particle_entropy": particle_entropy.cpu(),
        "particle_weights": weights.cpu(),
        "eta": float(eta), "lambda_h": float(lambda_h),
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


def softmax_weights(scores, eta):
    return F.softmax(eta * scores, dim=0)


# ---------------------------------------------------------------------------
# Scoring convenience helpers (shared by figs 2,3,5,6,8,9). They wrap the proven
# per-layer TestTimeLookaheadKVCluster scoring so every figure scores identically.
# ---------------------------------------------------------------------------
def get_repeated_kv(kvc):
    """GQA-repeat a layer's compact full K/V to query-head count -> ([1,H,n,D], [1,H,n,D])."""
    add_repo_root()
    from causal_lookahead_snapkv.monkeypatch.snapkv_utils import repeat_kv
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
    trajectory, and confidence-weighted multi scores (Eq. 13). Each is [H, n]."""
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
