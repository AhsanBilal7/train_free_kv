"""Token-ranking demo: CriticalKV vs AnDPro vs LORE-KV on the SAME captured state.

Why this script exists
----------------------
The end-to-end tables (scripts/run_h100_aces.sh) say LORE-KV wins; they do not say
*why*. This script produces the score-level evidence for the robustness claim: given
one prompt, one prefill and one set of lookahead particles, it scores every candidate
KV pair under each method's own criterion, selects a cache with a shared selector, and
measures how well each retained cache matches what the model actually needs while
answering -- the true decoding queries, recorded afterwards from a full-cache greedy
decode and used only as ground truth.

Criteria (all evaluated on identical captured K/V, window queries and particles)
-------------------------------------------------------------------------------
  criticalkv       repo `MODE=critical` (experiments/*/pred.py -> replace_*_adaptive):
                   pooled mean prompt-window attention over the last w queries,
                   maxpool(kernel=7) smoothed. Ported verbatim from
                   adaptive_snapkv/monkeypatch/snapkv_utils.py:AdaptiveSnapKVCluster.
  andpro           repo `MODE=andpro`: anchor-direction projection -- each chunk's
                   attention-weighted value vector projected on the summed droppable
                   value direction, accumulated over the w window queries. Ported
                   verbatim from andpro_snapkv/monkeypatch/snapkv_utils.py.
  lorekv_window    LORE-KV score at lambda_L = 0: projected leave-one-out deletion
                   effect (Eq. 5) on prompt-window queries only. This is the paper's
                   "AnDPro window" analysis criterion (App. mechanism protocol).
  lorekv           the full method (Eq. 7): window + lambda_L * sum_m omega_m * particle_m,
                   with omega from the config's (beta, gamma, lambda_D).
  oracle           the same projected score evaluated on the TRUE future decoding
                   queries. Not deployable -- it is the target every criterion is
                   trying to approximate.

Metrics (per sample x budget x criterion; per layer, then aggregated)
---------------------------------------------------------------------
  recall_at_b      |top_B(S) ∩ top_B(S*)| / B against the oracle ranking (Eq. 24)
  spearman         rank correlation with the oracle score
  fidelity         cos(o_retained, o_full) on the true future queries (Eq. 25)
  evidence_keep    fraction of gold-answer token positions surviving eviction
  evidence_enrich  P(answer token | retained) / P(answer token | prompt)
  overlap_*        pairwise top-B agreement between criteria (how far the rankings move)
  stability        LORE-KV top-B self-overlap between two disjoint halves of the
                   particle ensemble -- the sampling-noise robustness check

Selection is held FIXED across criteria (uniform per-head chunk top-B with the
protected set forced in), so every difference is attributable to the score alone;
this is the paper's own mechanism protocol. Budgets are in tokens and converted to
chunks with the config's chunk size, exactly as compress() does.

Outputs (everything under --out-dir, default ./demos)
----------------------------------------------------
  demos/<run>/manifest.json                        config, model, hyperparameters, git sha
  demos/<run>/samples/<dataset>__s<i>.npz          per-sample arrays (per layer, per budget)
  demos/<run>/summary_long.csv                     tidy rows: one metric value per row
  demos/<run>/summary.json                         aggregated mean / sem per (dataset, budget, method)
  demos/<run>/profiles/<dataset>__s<i>.npz         score profiles for the ranking figures

Usage (on a GPU node; login nodes have no GPU)
----------------------------------------------
  python scripts/demo_token_ranking.py --benchmark longbench --dataset single \\
      --samples-per-dataset 2 --budgets 128 512
  python scripts/demo_token_ranking.py --benchmark ruler --dataset 4096,8192 \\
      --samples-per-dataset 2 --budgets 128 512

  # both benchmarks through the cluster wrapper:
  sbatch scripts/run_demo_token_ranking.sh
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_ROOT, "experiments", "LongBench", "figure_making_script")
sys.path.insert(0, FIG_DIR)
sys.path.insert(0, REPO_ROOT)

import common  # experiments/LongBench/figure_making_script/common.py

METHODS = ["criticalkv", "andpro", "lorekv_window", "lorekv"]
ORACLE = "oracle"
PAIRS = [("criticalkv", "lorekv"), ("andpro", "lorekv"), ("criticalkv", "andpro"),
         ("lorekv_window", "lorekv")]


# ---------------------------------------------------------------------------
# Criterion 1: CriticalKV / MODE=critical  (adaptive_snapkv pooled window attention)
# ---------------------------------------------------------------------------
@torch.no_grad()
def score_criticalkv(kvc, key, kernel_size=7, pooling="maxpool"):
    """Port of AdaptiveSnapKVCluster.calcul_attn_sore -> [H, n] (window columns = -inf).

    attn over the last w prompt queries, causal inside the window, averaged over the
    query axis, then max-pooled along the candidate axis. The window columns are not
    scored there (they are protected), so they are returned as -inf and forced in by
    the shared selector.
    """
    q = kvc.window_queries[0].float()          # [H, w, D]
    k = key[0].float()                         # [H, n, D]
    H, w, D = q.shape
    n = k.shape[1]
    if n <= w:                     # nothing outside the protected window to score
        return torch.zeros((H, n), device=k.device)
    logits = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(D)      # [H, w, n]
    cw = min(w, n)
    mask = torch.full((cw, cw), float("-inf"), device=k.device)
    mc = torch.arange(cw, device=k.device)
    mask.masked_fill_(mc <= mc.view(-1, 1), 0.0)
    logits[:, -cw:, -cw:] = logits[:, -cw:, -cw:] + mask
    attn = torch.softmax(logits, dim=-1)                            # [H, w, n]
    a_mean = attn[:, :, : n - w].mean(dim=1)                        # [H, n-w]
    pool = F.max_pool1d if pooling == "maxpool" else F.avg_pool1d
    a_pool = pool(a_mean, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
    out = torch.full((H, n), float("-inf"), device=k.device)
    out[:, : n - w] = a_pool
    return out


# ---------------------------------------------------------------------------
# Criterion 2: AnDPro / MODE=andpro  (anchor-direction projection, chunk level)
# ---------------------------------------------------------------------------
@torch.no_grad()
def score_andpro(kvc, key, value, chunk_size=4, window_size=32, kernel_size=7,
                 pooling="maxpool"):
    """Port of AndProSnapKVCluster.update_kv_wo_gqa's scoring core -> [H, n].

    Per window query i, chunk-sum the attention-weighted values, take the inner product
    with the summed *droppable* chunk direction (the anchor), and accumulate over i.
    Chunk scores are expanded back to their tokens so every criterion is compared on
    the same [H, n] grid; the shared selector re-pools them to chunks anyway.
    """
    q = kvc.window_queries[0].float()          # [H, w, D]
    k = key[0].float()
    v = value[0].float()                       # [H, n, D]
    H, w, D = q.shape
    n = k.shape[1]
    logits = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(D)
    cw = min(w, n)
    mask = torch.full((cw, cw), float("-inf"), device=k.device)
    mc = torch.arange(cw, device=k.device)
    mask.masked_fill_(mc <= mc.view(-1, 1), 0.0)
    logits[:, -cw:, -cw:] = logits[:, -cw:, -cw:] + mask
    attn = torch.softmax(logits, dim=-1)                            # [H, w, n]
    pool = F.max_pool1d if pooling == "maxpool" else F.avg_pool1d
    attn = pool(attn, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)

    pad = (chunk_size - (n % chunk_size)) % chunk_size
    attn_p = F.pad(attn, (0, pad))                                  # [H, w, n+pad]
    v_p = F.pad(v, (0, 0, 0, pad))                                  # [H, n+pad, D]
    n_chunk = (n + pad) // chunk_size

    drop = torch.ones(n_chunk, dtype=torch.bool, device=k.device)
    drop[0] = False                                                 # attention sink
    drop[-max(1, window_size // chunk_size):] = False               # observation window
    dropf = drop.float().view(1, -1, 1)
    chunk_score = torch.zeros((H, n_chunk), device=k.device)
    # Per head, so the [w, n_chunk, D] temporary stays small on long contexts.
    for h in range(H):
        a_c = attn_p[h].view(w, n_chunk, chunk_size)                # [w, n_chunk, chunk]
        v_c = v_p[h].view(n_chunk, chunk_size, D)                   # [n_chunk, chunk, D]
        vw = torch.einsum("wct,ctd->wcd", a_c, v_c)                 # [w, n_chunk, D]
        anchor = (vw * dropf).sum(dim=1)                            # [w, D]  droppable direction
        chunk_score[h] = (vw * anchor.unsqueeze(1)).sum(dim=-1).sum(dim=0)
        del a_c, v_c, vw, anchor
    tok = chunk_score.repeat_interleave(chunk_size, dim=-1)[:, :n]  # [H, n]
    return tok


# ---------------------------------------------------------------------------
# Shared selector: uniform per-head chunk top-B with the protected set forced in.
# Mirrors TestTimeLookaheadKVCluster._select_chunks at tau = 0, lambda_k = lambda_v = 0.
# ---------------------------------------------------------------------------
@torch.no_grad()
def select_mask(scores, n, budget, window_size, chunk_size):
    """scores: [H, n] -> bool token mask [H, n] retaining `budget` tokens per head."""
    H = scores.shape[0]
    device = scores.device
    pad = (chunk_size - (n % chunk_size)) % chunk_size
    padded = F.pad(scores, (0, pad), value=float("-inf"))
    n_chunk = (n + pad) // chunk_size
    # amax, exactly as TestTimeLookaheadKVCluster._select_chunks pools a chunk
    chunk_score = padded.view(H, n_chunk, chunk_size).amax(dim=-1)

    protected = torch.zeros((H, n_chunk), dtype=torch.bool, device=device)
    protected[:, 0] = True
    protected[:, -max(1, window_size // chunk_size):] = True
    n_prot = int(protected[0].sum().item())

    keep_chunks = max(0, min(budget, n) // chunk_size - n_prot)
    s = chunk_score.masked_fill(protected, float("-inf"))
    keep = protected.clone()
    if keep_chunks > 0:
        idx = s.topk(min(keep_chunks, n_chunk - n_prot), dim=-1).indices
        keep.scatter_(1, idx, True)
    mask = keep.repeat_interleave(chunk_size, dim=-1)[:, :n]
    return mask


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float((ra @ rb) / den) if den > 0 else 0.0


@torch.no_grad()
def mask_overlap(mask_a, mask_b, restrict=None):
    """Mean per-head overlap |A ∩ B| / |A| (both masks keep the same count).

    `restrict` (bool [n]) limits the comparison to those positions -- used to drop the
    protected set, which every criterion retains by construction and which would
    otherwise account for ~25% of the agreement at B = 128.
    """
    if restrict is not None:
        mask_a = mask_a & restrict.unsqueeze(0)
        mask_b = mask_b & restrict.unsqueeze(0)
    inter = (mask_a & mask_b).sum(-1).float()
    denom = mask_a.sum(-1).float().clamp_min(1)
    return float((inter / denom).mean().item())


@torch.no_grad()
def fidelity(true_q, key, value, mask):
    """cos(o_retained, o_full) averaged over heads and future-query steps."""
    q = true_q[0].float()                     # [H, Tt, D]
    k = key[0].float()
    v = value[0].float()
    o_full = common.attn_outputs(q, k, v, None)
    o_keep = common.attn_outputs(q, k, v, mask)
    cos = F.cosine_similarity(o_full, o_keep, dim=-1)     # [H, Tt]
    return float(cos.mean().item())


def find_answer_positions(input_ids, answers, tokenizer):
    """Positions of gold-answer tokens in the prompt.

    Exact token-id subsequence match first (the paper's D4 rule); if an answer never
    appears as a contiguous id run (tokenizer boundary effects), fall back to the set of
    its content token ids, ignoring ids that occur in more than 2% of the prompt so a
    stopword does not swamp the class.
    """
    ids = input_ids[0].tolist()
    n = len(ids)
    hits = set()
    for ans in answers or []:
        if not isinstance(ans, str) or not ans.strip():
            continue
        for variant in (ans, " " + ans):
            a_ids = tokenizer(variant, add_special_tokens=False).input_ids
            if not a_ids:
                continue
            L = len(a_ids)
            found = False
            for s in range(0, n - L + 1):
                if ids[s:s + L] == a_ids:
                    hits.update(range(s, s + L))
                    found = True
            if found:
                break
        else:
            # Fallback ONLY for short answers (RULER needles, entity answers). For a long
            # free-form answer, scattered per-token matches are not "the evidence" -- on
            # narrativeqa the loose rule marked 5% of the prompt, which makes the
            # enrichment ratio meaningless. Those samples get an empty class instead and
            # drop out of the evidence metric.
            a_ids = tokenizer(ans, add_special_tokens=False).input_ids
            if len(a_ids) > 8:
                continue
            counts = {}
            for t in ids:
                counts[t] = counts.get(t, 0) + 1
            rare = {t for t in a_ids if counts.get(t, 0) <= max(1, int(0.005 * n))}
            hits.update(i for i, t in enumerate(ids) if t in rare)
    pos = np.array(sorted(hits), dtype=np.int64)
    if pos.size > max(1, int(0.02 * n)):
        return np.array([], dtype=np.int64)
    return pos


# ---------------------------------------------------------------------------
# Per-sample computation
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_sample(model, tokenizer, sample, cap, args, cfg):
    """Return (rows, per_sample_arrays, profiles) for one captured prompt."""
    from causal_lookahead_snapkv.monkeypatch.snapkv_utils import repeat_kv

    n = cap["n"]
    M = cap["M"]
    weights = cap["particle_weights"]
    lambda_L = cfg.lambda_lookahead
    chunk = cfg.chunk_size
    w = cfg.window_size
    layers = model.model.layers
    n_layers = len(layers)
    rep_layer = args.rep_layer if args.rep_layer >= 0 else n_layers // 2
    budgets = args.budgets

    ans_pos = find_answer_positions(sample["input_ids"], sample.get("answers"), tokenizer)
    device = sample["input_ids"].device
    ans_mask_t = torch.zeros(n, dtype=torch.bool, device=device)
    if ans_pos.size:
        ans_pos = ans_pos[ans_pos < n]
        ans_mask_t[torch.as_tensor(ans_pos, device=device)] = True
    n_answer = int(ans_mask_t.sum().item())

    all_methods = METHODS + [ORACLE]
    # [n_layers, n_budgets] accumulators, per method
    acc = {m: {k: np.full((n_layers, len(budgets)), np.nan, dtype=np.float64)
               for k in ("recall", "recall_cand", "fidelity", "evidence_keep")}
           for m in all_methods}
    # candidates = everything the protected set does not force in
    cand_mask = torch.ones(n, dtype=torch.bool, device=device)
    cand_mask[0] = False
    cand_mask[max(0, n - cfg.window_size):] = False
    rho = {m: np.full(n_layers, np.nan) for m in METHODS}
    ov = {f"{a}|{b}": np.full((n_layers, len(budgets)), np.nan) for a, b in PAIRS}
    stability = np.full((n_layers, len(budgets)), np.nan)
    profiles = {}

    for li, layer in enumerate(layers):
        kvc = layer.self_attn.kv_cluster
        if kvc.full_key_states is None:
            continue
        key = repeat_kv(kvc.full_key_states, kvc.num_key_value_groups)
        val = repeat_kv(kvc.full_value_states, kvc.num_key_value_groups)

        batches, flags = common.make_lookahead_batches(kvc, M)
        parts = kvc._project_and_score(key, val, batches, flags)         # [window, p_0..p_{M-1}]
        s_window = parts[0]
        s_lorekv = s_window.clone()
        for m in range(M):
            s_lorekv = s_lorekv + lambda_L * float(weights[m]) * parts[1 + m]

        scores = {
            "criticalkv": score_criticalkv(kvc, key, cfg.kernel_size, cfg.pooling),
            "andpro": score_andpro(kvc, key, val, chunk, w, cfg.kernel_size, cfg.pooling),
            "lorekv_window": s_window,
            "lorekv": s_lorekv,
        }
        true_q = cap["true_queries"][li] if cap["true_queries"] is not None else None
        if true_q is not None:
            scores[ORACLE] = common.oracle_scores(kvc, key, val, true_q)

        # sampling-noise robustness: two disjoint halves of the particle ensemble
        s_half = []
        if M >= 4:
            half = M // 2
            for grp in (range(half), range(half, M)):
                wsum = sum(float(weights[m]) for m in grp) or 1.0
                sh = s_window.clone()
                for m in grp:
                    sh = sh + lambda_L * (float(weights[m]) / wsum) * parts[1 + m]
                s_half.append(sh)

        oracle_head_mean = (scores[ORACLE].mean(0).cpu().numpy()
                            if ORACLE in scores else None)
        for meth in METHODS:
            if oracle_head_mean is not None:
                sm = scores[meth].mean(0).cpu().numpy()
                finite = sm[np.isfinite(sm)]
                sm = np.where(np.isfinite(sm), sm, finite.min() if finite.size else 0.0)
                rho[meth][li] = spearman(sm, oracle_head_mean)

        for bi, B in enumerate(budgets):
            masks = {m: select_mask(s, n, B, w, chunk) for m, s in scores.items()}
            if li == rep_layer:
                # fraction of heads retaining each position -> the retention-strip panel
                for meth, mk in masks.items():
                    profiles[f"keepfrac_b{B}_{meth}"] = (
                        mk.float().mean(0).cpu().numpy().astype(np.float32))
            for meth, mk in masks.items():
                if ORACLE in masks:
                    acc[meth]["recall"][li, bi] = mask_overlap(mk, masks[ORACLE])
                    acc[meth]["recall_cand"][li, bi] = mask_overlap(
                        mk, masks[ORACLE], restrict=cand_mask)
                if true_q is not None:
                    acc[meth]["fidelity"][li, bi] = fidelity(true_q, key, val, mk)
                if n_answer:
                    acc[meth]["evidence_keep"][li, bi] = float(
                        (mk[:, ans_mask_t].float().mean()).item())
            for a, b in PAIRS:
                ov[f"{a}|{b}"][li, bi] = mask_overlap(masks[a], masks[b])
            if s_half:
                m0 = select_mask(s_half[0], n, B, w, chunk)
                m1 = select_mask(s_half[1], n, B, w, chunk)
                stability[li, bi] = mask_overlap(m0, m1)

        if li == rep_layer:
            for meth, s in scores.items():
                profiles[f"score_{meth}"] = s.mean(0).float().cpu().numpy().astype(np.float32)
                profiles[f"headscore_{meth}"] = s.float().cpu().numpy().astype(np.float16)
            profiles["rep_layer"] = np.array(rep_layer)
            profiles["answer_positions"] = ans_pos
            profiles["n"] = np.array(n)
            profiles["budgets"] = np.array(budgets)
            profiles["window_size"] = np.array(w)
        del key, val, parts, scores
        torch.cuda.empty_cache()

    keep_frac = float(n_answer) / max(1, n)
    rows = []
    arrays = {
        "n": n, "n_answer": n_answer, "answer_positions": ans_pos,
        "budgets": np.array(budgets), "layers": np.arange(n_layers),
        "methods": np.array(all_methods), "rep_layer": rep_layer,
        "particle_weights": weights.numpy(), "M": M, "T": cap["T"],
        "dataset": sample["dataset"], "sample_idx": sample["sample_idx"],
        "context_length": sample["context_length"],
    }
    for meth in all_methods:
        for k, v in acc[meth].items():
            arrays[f"{meth}__{k}"] = v
        if meth in rho:
            arrays[f"{meth}__spearman"] = rho[meth]
    for k, v in ov.items():
        arrays["overlap__" + k.replace("|", "_vs_")] = v
    arrays["lorekv__stability"] = stability

    for bi, B in enumerate(budgets):
        for meth in all_methods:
            base = dict(dataset=sample["dataset"], sample=sample["sample_idx"],
                        context_length=sample["context_length"], budget=B, method=meth)
            for k in ("recall", "recall_cand", "fidelity", "evidence_keep"):
                v = np.nanmean(acc[meth][k][:, bi])
                if np.isfinite(v):
                    rows.append({**base, "metric": k, "value": float(v)})
            if meth in rho and np.isfinite(np.nanmean(rho[meth])):
                rows.append({**base, "metric": "spearman", "value": float(np.nanmean(rho[meth]))})
            if n_answer:
                ek = np.nanmean(acc[meth]["evidence_keep"][:, bi])
                kept_frac = float(min(B, n)) / max(1, n)
                if np.isfinite(ek) and kept_frac > 0:
                    rows.append({**base, "metric": "evidence_enrich",
                                 "value": float(ek / kept_frac)})
        for a, b in PAIRS:
            v = np.nanmean(ov[f"{a}|{b}"][:, bi])
            if np.isfinite(v):
                rows.append(dict(dataset=sample["dataset"], sample=sample["sample_idx"],
                                 context_length=sample["context_length"], budget=B,
                                 method=f"{a}_vs_{b}", metric="topb_overlap", value=float(v)))
        v = np.nanmean(stability[:, bi])
        if np.isfinite(v):
            rows.append(dict(dataset=sample["dataset"], sample=sample["sample_idx"],
                             context_length=sample["context_length"], budget=B,
                             method="lorekv", metric="particle_stability", value=float(v)))
    arrays["answer_token_fraction"] = keep_frac
    return rows, arrays, profiles


# ---------------------------------------------------------------------------
def git_sha():
    try:
        return subprocess.check_output(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def aggregate(rows):
    """mean / sem / n per (dataset, budget, method, metric)."""
    agg = {}
    for r in rows:
        k = (r["dataset"], r["budget"], r["method"], r["metric"])
        agg.setdefault(k, []).append(r["value"])
    out = []
    for (ds, b, meth, metric), vals in sorted(agg.items(), key=lambda x: str(x[0])):
        a = np.asarray(vals, dtype=np.float64)
        out.append(dict(dataset=ds, budget=int(b), method=meth, metric=metric,
                        mean=float(a.mean()), sem=float(a.std(ddof=1) / math.sqrt(len(a)))
                        if len(a) > 1 else 0.0, n=len(a)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "config", "lorekv-ent-unif-topk.conf"),
                    help="LORE-KV config; supplies M, T, lambda_L, (beta, gamma, lambda_D), chunk size")
    ap.add_argument("--model", default=None, help="override MODEL_PATH from the config")
    ap.add_argument("--benchmark", default="longbench", choices=["longbench", "ruler"])
    ap.add_argument("--dataset", default="single",
                    help="longbench: single|full|comma list; ruler: 4096|8192|16384|comma list")
    ap.add_argument("--dataset-path", default=None)
    ap.add_argument("--samples-per-dataset", type=int, default=2)
    ap.add_argument("--budgets", type=int, nargs="+", default=[128, 512])
    ap.add_argument("--decode-steps", type=int, default=24,
                    help="true future queries recorded from a full-cache greedy decode")
    ap.add_argument("--max-length", type=int, default=32000)
    ap.add_argument("--rep-layer", type=int, default=-1, help="-1 = middle layer")
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "demos"))
    ap.add_argument("--run-name", default=None, help="subfolder name (default: auto tag)")
    ap.add_argument("--no-profiles", action="store_true", help="skip the per-layer score dumps")
    args = ap.parse_args()

    conf = common.parse_conf(args.config)
    model_path = args.model or conf.get("MODEL_PATH", "mistralai/Mistral-7B-Instruct-v0.2")
    run = args.run_name or "__".join([
        common._sanitize(os.path.splitext(os.path.basename(args.config))[0]),
        common._sanitize(str(model_path).rstrip("/").split("/")[-1]),
        common._sanitize(args.benchmark), common._sanitize(args.dataset),
        "b" + "-".join(str(b) for b in args.budgets),
    ])
    out_dir = os.path.join(args.out_dir, run)
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
    if not args.no_profiles:
        os.makedirs(os.path.join(out_dir, "profiles"), exist_ok=True)
    print(f"[demo] writing to {out_dir}", flush=True)

    common.apply_lookahead_patch()
    model, tokenizer = common.load_model_and_tokenizer(model_path)
    common.configure_model(model, conf, max(args.budgets))
    cfg = model.model.config

    manifest = dict(
        run=run, created=time.strftime("%Y-%m-%dT%H:%M:%S"), git_sha=git_sha(),
        config=os.path.relpath(args.config, REPO_ROOT), model=model_path,
        benchmark=args.benchmark, dataset=args.dataset,
        samples_per_dataset=args.samples_per_dataset, budgets=args.budgets,
        decode_steps=args.decode_steps, max_length=args.max_length,
        methods=METHODS + [ORACLE],
        hyperparams=dict(window_size=cfg.window_size, chunk_size=cfg.chunk_size,
                         num_particles=cfg.num_particles, lookahead_len=cfg.lookahead_len,
                         lambda_lookahead=cfg.lambda_lookahead,
                         beta=cfg.particle_beta, gamma=cfg.particle_gamma,
                         lambda_d=cfg.lambda_d,
                         robust_alpha=cfg.robust_alpha, robust_topk_beta=cfg.robust_topk_beta,
                         score_objective=getattr(cfg, "score_objective", "projected")),
        selection="uniform per-head chunk top-B, protected = {sink} u window (shared by all criteria)",
    )
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    rows = []
    n_done = 0
    for sample in common.load_samples(
        args.benchmark, tokenizer, args.dataset, model_path, dataset_path=args.dataset_path,
        max_length=args.max_length, samples_per_dataset=args.samples_per_dataset,
    ):
        t0 = time.time()
        cap = common.collect_and_lookahead(
            model, sample["input_ids"], sample["attention_mask"],
            decode_steps=args.decode_steps, capture_true=True)
        if cap.get("true_queries") is None and not cap.get("per_layer"):
            print(f"[skip] {sample['dataset']}#{sample['sample_idx']} (OOM sentinel)", flush=True)
            common.free_sample(model, cap)
            continue
        srows, arrays, profiles = run_sample(model, tokenizer, sample, cap, args, cfg)
        rows.extend(srows)
        tag = f"{common._sanitize(sample['dataset'])}__s{sample['sample_idx']}"
        np.savez_compressed(os.path.join(out_dir, "samples", tag + ".npz"), **arrays)
        if profiles and not args.no_profiles:
            np.savez_compressed(os.path.join(out_dir, "profiles", tag + ".npz"), **profiles)
        n_done += 1
        got = {r["method"]: r["value"] for r in srows
               if r["metric"] == "recall" and r["budget"] == args.budgets[0]}
        print(f"[{sample['dataset']}#{sample['sample_idx']}] n={cap['n']} "
              f"{time.time() - t0:.0f}s  recall@{args.budgets[0]}: "
              + "  ".join(f"{k}={v:.3f}" for k, v in got.items()), flush=True)
        common.free_sample(model, cap)

        with open(os.path.join(out_dir, "summary_long.csv"), "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=["dataset", "sample", "context_length",
                                                "budget", "method", "metric", "value"])
            wtr.writeheader()
            wtr.writerows(rows)
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(dict(manifest=manifest, aggregate=aggregate(rows)), f, indent=2)

    print(f"[demo] {n_done} samples -> {out_dir}", flush=True)
    for a in aggregate(rows):
        if a["metric"] in ("recall", "evidence_enrich"):
            print(f"  {a['dataset']:<16} B={a['budget']:<5} {a['method']:<16} "
                  f"{a['metric']:<16} {a['mean']:.4f} ± {a['sem']:.4f}  (n={a['n']})")


if __name__ == "__main__":
    main()
