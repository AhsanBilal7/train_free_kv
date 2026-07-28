"""Figure 8 (compute stage) - Layer-wise and head-wise structure of the scores.

  A. layer perturbation energy   E_l   = sum_h sum_i S_i^{l,h}   (LORE-KV multi score)
  B. head future-query gain      G_lh  = recall@B(S_multi vs S*) - recall@B(S_window vs S*)
                                         per (layer, head): where does adding future
                                         queries actually improve retention?
  C. head query mismatch         MM_lh = 1 - align(window queries, true queries) per head

Saved: layer_head_data.npz. Run on the GPU cluster, e.g.:
  python compute_layer_head.py --config config/tlckv-pe-m4-adap-topk.conf \
         --dataset single --samples-per-dataset 3 --budget 128
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common


def _unit(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


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
    ap.add_argument("--decode-steps", type=int, default=24)
    ap.add_argument("--max-length", type=int, default=64000)
    ap.add_argument("--out", default=None, help="explicit .npz path (overrides the auto tagged name)")
    ap.add_argument("--out-dir", default=None, help="dir for the auto-named tagged .npz (default: this folder)")
    args = ap.parse_args()

    conf = common.parse_conf(args.config)
    model_path = args.model or conf.get("MODEL_PATH", "mistralai/Mistral-7B-Instruct-v0.2")
    args.out = common.default_npz_path(
        args.out, args.out_dir, os.path.dirname(os.path.abspath(__file__)), "layer_head_data",
        common.build_tag(args.config, model_path, args.budget, args.dataset))
    ckpt = common.Checkpoint(args.out)
    if os.path.exists(args.out) and not os.path.exists(ckpt.path):
        print(f"{args.out} already complete; skipping recompute"); return
    common.apply_lookahead_patch()
    model, tokenizer = common.load_model_and_tokenizer(model_path)
    common.configure_model(model, conf, args.budget)
    lambda_L = model.model.config.lambda_lookahead
    n_layers = len(model.model.layers)

    state, done = ckpt.load_or({"E_l": np.zeros(n_layers), "E_cnt": np.zeros(n_layers),
                               "G_lh": None, "MM_lh": None, "count_lh": None})
    E_l = state["E_l"]; E_cnt = state["E_cnt"]
    G_lh = state["G_lh"]; MM_lh = state["MM_lh"]; count_lh = state["count_lh"]

    def _flush():
        ckpt.flush({"E_l": E_l, "E_cnt": E_cnt, "G_lh": G_lh,
                    "MM_lh": MM_lh, "count_lh": count_lh}, done)

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
        M = cap["M"]; weights = cap["particle_weights"]

        for li, layer in enumerate(model.model.layers):
            kvc = layer.self_attn.kv_cluster
            tq = cap["true_queries"][li]
            if kvc.full_key_states is None or tq is None:
                continue
            key, val = common.get_repeated_kv(kvc)
            H = key.shape[1]
            if G_lh is None:
                G_lh = np.zeros((n_layers, H)); MM_lh = np.zeros((n_layers, H)); count_lh = np.zeros((n_layers, H))

            batches, flags = common.make_lookahead_batches(kvc, M)
            scores = kvc._project_and_score(key, val, batches, flags)
            s_win, _, s_multi = common.combine_lookahead_scores(scores, weights, lambda_L, M)
            oracle = common.oracle_scores(kvc, key, val, tq)

            E_l[li] += float(s_multi.clamp_min(0).sum().item()); E_cnt[li] += 1

            sm = s_multi.cpu().numpy(); sw = s_win.cpu().numpy(); orc = oracle.cpu().numpy()
            for h in range(H):
                G_lh[li, h] += (common.recall_at_b(sm[h], orc[h], args.budget)
                                - common.recall_at_b(sw[h], orc[h], args.budget))

            # per-head window-vs-true mismatch
            tqh = _unit(tq[0].float()); winh = _unit(kvc.window_queries[0].float())
            sim = torch.matmul(tqh, winh.transpose(1, 2))       # [H,Tt,w]
            cov_h = sim.max(dim=-1).values.mean(dim=-1)          # [H]
            MM_lh[li] += (1.0 - cov_h).cpu().numpy()
            count_lh[li] += 1

        done.add(key); _flush()
        print(f"[{sample['dataset']}#{sample['sample_idx']}] n={cap['n']} layer/head accumulated", flush=True)
        common.free_sample(model, cap)

    E_l = E_l / np.maximum(E_cnt, 1)
    G_lh = (G_lh if G_lh is not None else np.zeros((n_layers, 1))) / np.maximum(count_lh if count_lh is not None else 1, 1)
    MM_lh = (MM_lh if MM_lh is not None else np.zeros((n_layers, 1))) / np.maximum(count_lh if count_lh is not None else 1, 1)
    np.savez(args.out, budget=args.budget, dataset=args.dataset,
             E_layer=E_l, G_head=G_lh, MM_head=MM_lh)
    ckpt.done()
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
