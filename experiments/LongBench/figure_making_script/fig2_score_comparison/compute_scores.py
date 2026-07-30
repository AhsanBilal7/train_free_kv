"""Figure 2 (compute stage) - Raw token-importance score comparison.

Three scoring rules over the same candidate KV pairs, using the same captured
prompt + lookahead particles:

  * attention        - simple mean prompt-window attention weight a_{q,i}
                       (the SnapKV/H2O-style importance signal).
  * single_lookahead - LORE-KV output-projection perturbation score with the
                       window queries + ONE lookahead trajectory (Eq. 13, M=1).
  * lorekv_multi     - LORE-KV score with window + M confidence-weighted
                       trajectories (Eq. 13, the full method).

We reuse the *exact* per-layer scoring already implemented on each layer's
`kv_cluster` (TestTimeLookaheadKVCluster._project_and_score), so the numbers
match what the eviction actually uses.

Saved: score_data.npz (read by plot_scores.py).

Run on the GPU cluster, e.g.:
  python compute_scores.py --config config/tlckv-pe-m4-adap-topk.conf \
         --dataset single --samples-per-dataset 3 --budget 128
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    denom = (np.linalg.norm(ra) * np.linalg.norm(rb))
    return float((ra @ rb) / denom) if denom > 0 else 0.0


def _topb_overlap(a, b, B):
    B = min(B, len(a))
    ta = set(np.argsort(-a)[:B].tolist())
    tb = set(np.argsort(-b)[:B].tolist())
    return len(ta & tb) / B


@torch.no_grad()
def attention_score_layer(kvc, key_states):
    """Mean prompt-window attention weight per candidate token, averaged over heads -> [n]."""
    import math
    q = kvc.window_queries[0].float()      # [H, w, D]
    k = key_states[0].float()              # [H, n, D]
    H, w, D = q.shape
    n = k.shape[1]
    out = torch.zeros(H, n, device=k.device)
    for h in range(H):
        logits = (q[h] @ k[h].T) / math.sqrt(D)        # [w, n]
        mask = torch.full((w, w), float("-inf"), device=k.device)
        mc = torch.arange(w, device=k.device)
        mask.masked_fill_(mc <= mc.view(-1, 1), 0.0)
        logits[:, -w:] = logits[:, -w:] + mask
        a = torch.softmax(logits, dim=-1)              # [w, n]
        out[h] = a.mean(0)
    return out.mean(0).cpu().numpy()                   # [n]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(common.CONFIG_DIR, "tlckv-pe-m4-adap-topk.conf"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--benchmark", default="longbench", choices=["longbench", "ruler"])
    ap.add_argument("--dataset", default="single")
    ap.add_argument("--dataset-path", default=None)
    ap.add_argument("--samples-per-dataset", type=int, default=3)
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--rep-layer", type=int, default=-1, help="layer whose per-token profile is saved (-1=middle)")
    ap.add_argument("--max-length", type=int, default=64000)
    ap.add_argument("--out", default=None, help="explicit .npz path (overrides the auto tagged name)")
    ap.add_argument("--out-dir", default=None, help="dir for the auto-named tagged .npz (default: this folder)")
    args = ap.parse_args()

    common.add_repo_root()
    from causal_lookahead_snapkv.monkeypatch.snapkv_utils import repeat_kv

    conf = common.parse_conf(args.config)
    model_path = args.model or conf.get("MODEL_PATH", "mistralai/Mistral-7B-Instruct-v0.2")
    args.out = common.default_npz_path(
        args.out, args.out_dir, os.path.dirname(os.path.abspath(__file__)), "score_data",
        common.build_tag(args.config, model_path, args.budget, args.dataset))
    ckpt = common.Checkpoint(args.out)
    if os.path.exists(args.out) and not os.path.exists(ckpt.path):
        print(f"{args.out} already complete; skipping recompute"); return
    common.apply_lookahead_patch()
    model, tokenizer = common.load_model_and_tokenizer(model_path)
    common.configure_model(model, conf, args.budget)
    lambda_L = model.model.config.lambda_lookahead
    M = model.model.config.num_particles

    methods = ["attention", "single_lookahead", "lorekv_multi"]
    pairs = [("attention", "single_lookahead"), ("attention", "lorekv_multi"), ("single_lookahead", "lorekv_multi")]
    state, done = ckpt.load_or({"spearman": {f"{a}|{b}": [] for a, b in pairs},
                               "overlap": {f"{a}|{b}": [] for a, b in pairs},
                               "rep_profile": None, "rep_raw": None})
    spearman = state["spearman"]; overlap = state["overlap"]
    rep_profile = state["rep_profile"]; rep_raw = state["rep_raw"]

    def _flush():
        ckpt.flush({"spearman": spearman, "overlap": overlap,
                    "rep_profile": rep_profile, "rep_raw": rep_raw}, done)

    for sample in common.load_samples(
        args.benchmark, tokenizer, args.dataset, model_path, dataset_path=args.dataset_path,
        max_length=args.max_length, samples_per_dataset=args.samples_per_dataset,
    ):
        key = common.entry_key(sample)
        if key in done:
            continue
        cap = common.collect_and_lookahead(
            model, sample["input_ids"], sample["attention_mask"], decode_steps=0, capture_true=False,
        )
        weights = cap["particle_weights"]
        n_layers = len(model.model.layers)
        rep_layer = args.rep_layer if args.rep_layer >= 0 else n_layers // 2

        for li, layer in enumerate(model.model.layers):
            kvc = layer.self_attn.kv_cluster
            if kvc.full_key_states is None:
                continue
            k_full = repeat_kv(kvc.full_key_states, kvc.num_key_value_groups)
            val = repeat_kv(kvc.full_value_states, kvc.num_key_value_groups)
            batches = [kvc.window_queries]
            flags = [True]
            for m in range(M):
                buf = kvc._lookahead_buffers[m]
                q_m = torch.cat(buf, dim=2) if buf else kvc.window_queries[:, :, :0, :]
                batches.append(q_m); flags.append(False)

            scores = kvc._project_and_score(k_full, val, batches, flags)   # list [H, n]
            s_win = scores[0]
            s_single = s_win + lambda_L * scores[1]
            s_multi = s_win.clone()
            for m in range(M):
                s_multi = s_multi + lambda_L * float(weights[m]) * scores[1 + m]

            m_scores = {
                "attention": attention_score_layer(kvc, k_full),
                "single_lookahead": s_single.mean(0).cpu().numpy(),
                "lorekv_multi": s_multi.mean(0).cpu().numpy(),
            }
            for a, b in pairs:
                spearman[f"{a}|{b}"].append(_spearman(m_scores[a], m_scores[b]))
                overlap[f"{a}|{b}"].append(_topb_overlap(m_scores[a], m_scores[b], args.budget))

            if rep_profile is None and li == rep_layer:
                rep_profile = {k: np.sort(v)[::-1].copy() for k, v in m_scores.items()}
                rep_raw = {k: v.copy() for k, v in m_scores.items()}

        done.add(key); _flush()
        print(f"[{sample['dataset']}#{sample['sample_idx']}] n={cap['n']} "
              f"overlap(attn,multi)~{np.mean(overlap['attention|lorekv_multi']):.3f}", flush=True)
        common.free_sample(model, cap)

    save = dict(
        methods=np.array(methods), budget=args.budget, dataset=args.dataset,
        pairs=np.array([f"{a}|{b}" for a, b in pairs]),
        spearman_mean=np.array([np.mean(spearman[f"{a}|{b}"]) for a, b in pairs]),
        spearman_std=np.array([np.std(spearman[f"{a}|{b}"]) for a, b in pairs]),
        overlap_mean=np.array([np.mean(overlap[f"{a}|{b}"]) for a, b in pairs]),
        overlap_std=np.array([np.std(overlap[f"{a}|{b}"]) for a, b in pairs]),
    )
    if rep_profile is not None:
        for k in methods:
            save[f"sorted_{k}"] = rep_profile[k]
            save[f"raw_{k}"] = rep_raw[k]
    np.savez(args.out, **save)
    ckpt.done()
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
