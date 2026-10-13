"""Figure 3 (compute stage) - Full-cache vs evicted-cache attention-output distribution.

The claim behind output-aware eviction is that keeping the right KV pairs preserves
what the model actually computes at decode time. We test this directly: for the TRUE
full-cache decode queries, we compare the attention output o_q produced by

  * the FULL cache,
  * the cache evicted by LORE-KV (window + confidence-weighted lookahead perturbation),
  * the cache evicted by a plain attention (SnapKV-style) score,

all at the same budget B and same per-head budget. We record, per (layer, head,
query), the cosine similarity and relative L2 error of the evicted output vs the
full output. A method that preserves the decode-time distribution keeps cosine near 1.

Saved: distribution_data.npz (read by plot_distribution.py).

Run on the GPU cluster, e.g.:
  python compute_distribution.py --config config/lorekv-ent-adap-topk.conf \
         --dataset single --samples-per-dataset 2 --budget 128 --decode-steps 16
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common


@torch.no_grad()
def head_topb_mask(scores, n, budget, window_size):
    """scores: [H, n] -> bool mask [H, n] keeping the first token, the last `window_size`
    tokens, and the top (budget - protected) candidates per head."""
    H = scores.shape[0]
    device = scores.device
    protected = torch.zeros(H, n, dtype=torch.bool, device=device)
    protected[:, 0] = True
    protected[:, max(0, n - window_size):] = True
    s = scores.clone()
    s[protected] = float("inf")                       # force-keep protected
    b = min(budget, n)
    idx = s.topk(b, dim=-1).indices
    mask = torch.zeros(H, n, dtype=torch.bool, device=device)
    mask.scatter_(1, idx, True)
    return mask


@torch.no_grad()
def attn_outputs(q, k, v, mask=None):
    """q:[H,Tt,D] k,v:[H,n,D] mask:[H,n] or None -> o:[H,Tt,D]."""
    D = q.shape[-1]
    logits = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(D)   # [H,Tt,n]
    if mask is not None:
        logits = logits.masked_fill(~mask.unsqueeze(1), float("-inf"))
    a = F.softmax(logits, dim=-1)
    return torch.matmul(a, v)                                     # [H,Tt,D]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(common.CONFIG_DIR, "lorekv-ent-adap-topk.conf"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--benchmark", default="longbench", choices=["longbench", "ruler"])
    ap.add_argument("--dataset", default="single")
    ap.add_argument("--dataset-path", default=None)
    ap.add_argument("--samples-per-dataset", type=int, default=2)
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--decode-steps", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=64000)
    ap.add_argument("--out", default=None, help="explicit .npz path (overrides the auto tagged name)")
    ap.add_argument("--out-dir", default=None, help="dir for the auto-named tagged .npz (default: this folder)")
    args = ap.parse_args()

    common.add_repo_root()
    from lorekv.monkeypatch.scoring import repeat_kv

    conf = common.parse_conf(args.config)
    model_path = args.model or conf.get("MODEL_PATH", "mistralai/Mistral-7B-Instruct-v0.2")
    args.out = common.default_npz_path(
        args.out, args.out_dir, os.path.dirname(os.path.abspath(__file__)), "distribution_data",
        common.build_tag(args.config, model_path, args.budget, args.dataset))
    ckpt = common.Checkpoint(args.out)
    if os.path.exists(args.out) and not os.path.exists(ckpt.path):
        print(f"{args.out} already complete; skipping recompute"); return
    common.apply_lookahead_patch()
    model, tokenizer = common.load_model_and_tokenizer(model_path)
    common.configure_model(model, conf, args.budget)
    lambda_L = model.model.config.lambda_lookahead
    window_size = model.model.config.window_size
    M = model.model.config.num_particles

    n_layers = len(model.model.layers)
    state, done = ckpt.load_or({"cos_lorekv": [], "cos_attn": [], "rel_lorekv": [], "rel_attn": [],
                               "pl_lk": [[] for _ in range(n_layers)],
                               "pl_at": [[] for _ in range(n_layers)]})
    cos_lorekv = state["cos_lorekv"]; cos_attn = state["cos_attn"]
    rel_lorekv = state["rel_lorekv"]; rel_attn = state["rel_attn"]
    per_layer_cos_lorekv = state["pl_lk"]; per_layer_cos_attn = state["pl_at"]

    def _flush():
        ckpt.flush({"cos_lorekv": cos_lorekv, "cos_attn": cos_attn,
                    "rel_lorekv": rel_lorekv, "rel_attn": rel_attn,
                    "pl_lk": per_layer_cos_lorekv, "pl_at": per_layer_cos_attn}, done)

    for sample in common.load_samples(
        args.benchmark, tokenizer, args.dataset, model_path, dataset_path=args.dataset_path,
        max_length=args.max_length, samples_per_dataset=args.samples_per_dataset,
    ):
        key = common.entry_key(sample)
        if key in done:
            continue
        cap = common.collect_and_lookahead(
            model, sample["input_ids"], sample["attention_mask"],
            decode_steps=args.decode_steps, capture_true=True,
        )
        if cap["true_queries"] is None:
            common.free_sample(model, cap)
            done.add(key); _flush()
            continue
        weights = cap["particle_weights"]

        for li, layer in enumerate(model.model.layers):
            kvc = layer.self_attn.kv_cluster
            tq = cap["true_queries"][li]
            if kvc.full_key_states is None or tq is None:
                continue
            k_full = repeat_kv(kvc.full_key_states, kvc.num_key_value_groups)[0]   # [H,n,D]
            val = repeat_kv(kvc.full_value_states, kvc.num_key_value_groups)[0]
            q = tq[0]                                                            # [H,Tt,D]
            H, n, D = k_full.shape

            key_b = k_full.unsqueeze(0); val_b = val.unsqueeze(0)
            batches = [kvc.window_queries]
            flags = [True]
            for m in range(M):
                buf = kvc._lookahead_buffers[m]
                q_m = torch.cat(buf, dim=2) if buf else kvc.window_queries[:, :, :0, :]
                batches.append(q_m); flags.append(False)
            scores = kvc._project_and_score(key_b, val_b, batches, flags)       # list [H,n]

            s_multi = scores[0].clone()
            for m in range(M):
                s_multi = s_multi + lambda_L * float(weights[m]) * scores[1 + m]
            # attention-only importance (SnapKV-style): mean window attention weight
            qa = kvc.window_queries[0].float()                                  # [H,w,D]
            aw_logits = torch.matmul(qa, k_full.transpose(1, 2).float()) / math.sqrt(D)
            s_attn = F.softmax(aw_logits, dim=-1).mean(1)                       # [H,n]

            mask_lorekv = head_topb_mask(s_multi, n, args.budget, window_size)
            mask_attn = head_topb_mask(s_attn, n, args.budget, window_size)

            q = q.float()
            o_full = attn_outputs(q, k_full.float(), val.float())
            o_lk = attn_outputs(q, k_full.float(), val.float(), mask_lorekv)
            o_at = attn_outputs(q, k_full.float(), val.float(), mask_attn)

            def cos_rel(o_ev):
                c = F.cosine_similarity(o_ev, o_full, dim=-1).flatten()          # [H*Tt]
                r = (o_ev - o_full).norm(dim=-1).flatten() / (o_full.norm(dim=-1).flatten() + 1e-8)
                return c.cpu().numpy(), r.cpu().numpy()

            c_lk, r_lk = cos_rel(o_lk)
            c_at, r_at = cos_rel(o_at)
            cos_lorekv.append(c_lk); rel_lorekv.append(r_lk)
            cos_attn.append(c_at); rel_attn.append(r_at)
            per_layer_cos_lorekv[li].append(float(c_lk.mean()))
            per_layer_cos_attn[li].append(float(c_at.mean()))

        done.add(key); _flush()
        print(f"[{sample['dataset']}#{sample['sample_idx']}] n={cap['n']} "
              f"meanCos LORE-KV={np.concatenate(cos_lorekv).mean():.4f} attn={np.concatenate(cos_attn).mean():.4f}",
              flush=True)
        common.free_sample(model, cap)

    np.savez(
        args.out,
        budget=args.budget, dataset=args.dataset, decode_steps=args.decode_steps,
        cos_lorekv=np.concatenate(cos_lorekv), cos_attn=np.concatenate(cos_attn),
        rel_lorekv=np.concatenate(rel_lorekv), rel_attn=np.concatenate(rel_attn),
        per_layer_cos_lorekv=np.array([np.mean(x) if x else np.nan for x in per_layer_cos_lorekv]),
        per_layer_cos_attn=np.array([np.mean(x) if x else np.nan for x in per_layer_cos_attn]),
    )
    ckpt.done()
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
