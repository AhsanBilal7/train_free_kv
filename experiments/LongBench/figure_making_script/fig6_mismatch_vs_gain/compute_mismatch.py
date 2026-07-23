"""Figure 6 (compute stage) - Query-mismatch vs improvement (per-example causal case).

Tests the predicted condition under which LORE-KV should help: the gain should grow
with how much the prompt-window queries differ from the true decoding queries.

Per sample:
  mismatch  M(X) = 1 - alignment(Q_window, Q_true)          (layer-averaged)
  gains (LORE-KV multi over a window-only / AnDPro-like proxy, and over attention):
    * oracle recall gain   : recall@B(S_multi, S*) - recall@B(S_window, S*)
    * output-fidelity gain : cos(o_LORE, o_full) - cos(o_window, o_full)
                             cos(o_LORE, o_full) - cos(o_attn,   o_full)

Saved: mismatch_data.npz. Run on the GPU cluster, e.g.:
  python compute_mismatch.py --config config/tlckv-pe-m4-adap-topk.conf \
         --dataset single --samples-per-dataset 8 --budget 128
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


def _cos_to_full(q, key, val, mask):
    o_full = common.attn_outputs(q, key, val)
    o_ev = common.attn_outputs(q, key, val, mask)
    return float(F.cosine_similarity(o_ev, o_full, dim=-1).mean().item())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(common.CONFIG_DIR, "tlckv-pe-m4-adap-topk.conf"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--benchmark", default="longbench", choices=["longbench", "ruler"])
    ap.add_argument("--dataset", default="single")
    ap.add_argument("--dataset-path", default=None)
    ap.add_argument("--samples-per-dataset", type=int, default=8)
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--decode-steps", type=int, default=24)
    ap.add_argument("--max-length", type=int, default=64000)
    ap.add_argument("--out", default=None, help="explicit .npz path (overrides the auto tagged name)")
    ap.add_argument("--out-dir", default=None, help="dir for the auto-named tagged .npz (default: this folder)")
    args = ap.parse_args()

    conf = common.parse_conf(args.config)
    model_path = args.model or conf.get("MODEL_PATH", "mistralai/Mistral-7B-Instruct-v0.2")
    args.out = common.default_npz_path(
        args.out, args.out_dir, os.path.dirname(os.path.abspath(__file__)), "mismatch_data",
        common.build_tag(args.config, model_path, args.budget, args.dataset))
    common.apply_lookahead_patch()
    model, tokenizer = common.load_model_and_tokenizer(model_path)
    common.configure_model(model, conf, args.budget)
    eta = model.model.config.particle_eta
    lambda_L = model.model.config.lambda_lookahead
    window_size = model.model.config.window_size

    mismatch, g_recall, g_fid_win, g_fid_attn, labels = [], [], [], [], []

    for sample in common.load_samples(
        args.benchmark, tokenizer, args.dataset, model_path, dataset_path=args.dataset_path,
        max_length=args.max_length, samples_per_dataset=args.samples_per_dataset,
    ):
        cap = common.collect_and_lookahead(
            model, sample["input_ids"], sample["attention_mask"],
            decode_steps=args.decode_steps, capture_true=True,
        )
        if cap["true_queries"] is None:
            common.cleanup_clusters(model); continue
        M = cap["M"]; weights = cap["particle_weights"]

        cov_win, rec_gain, fid_win, fid_attn = [], [], [], []
        for li, layer in enumerate(model.model.layers):
            kvc = layer.self_attn.kv_cluster
            tq = cap["true_queries"][li]
            if kvc.full_key_states is None or tq is None:
                continue
            key, val = common.get_repeated_kv(kvc)
            batches, flags = common.make_lookahead_batches(kvc, M)
            scores = kvc._project_and_score(key, val, batches, flags)
            s_win, _, s_multi = common.combine_lookahead_scores(scores, weights, lambda_L, M)
            oracle = common.oracle_scores(kvc, key, val, tq)
            n = key.shape[2]

            # mismatch: window-query coverage of the true queries
            cov_win.append(common.weighted_coverage(tq[0], [kvc.window_queries[0]], torch.ones(1)))

            # oracle recall gain (head-averaged scores)
            om = oracle.mean(0).cpu().numpy()
            rec_gain.append(common.recall_at_b(s_multi.mean(0).cpu().numpy(), om, args.budget)
                            - common.recall_at_b(s_win.mean(0).cpu().numpy(), om, args.budget))

            # attention (SnapKV-style) importance
            qa = kvc.window_queries[0].float()
            aw = torch.matmul(qa, key[0].transpose(1, 2).float()) / math.sqrt(key.shape[-1])
            s_attn = F.softmax(aw, dim=-1).mean(1)              # [H,n]

            q = tq[0].float(); kf = key[0].float(); vf = val[0].float()
            m_lk = common.head_topb_mask(s_multi, n, args.budget, window_size)
            m_wn = common.head_topb_mask(s_win, n, args.budget, window_size)
            m_at = common.head_topb_mask(s_attn, n, args.budget, window_size)
            c_lk = _cos_to_full(q, kf, vf, m_lk)
            fid_win.append(c_lk - _cos_to_full(q, kf, vf, m_wn))
            fid_attn.append(c_lk - _cos_to_full(q, kf, vf, m_at))

        if not cov_win:
            common.cleanup_clusters(model); continue
        mismatch.append(1.0 - float(np.mean(cov_win)))
        g_recall.append(float(np.mean(rec_gain)))
        g_fid_win.append(float(np.mean(fid_win)))
        g_fid_attn.append(float(np.mean(fid_attn)))
        labels.append(sample["dataset"])
        print(f"[{sample['dataset']}#{sample['sample_idx']}] mismatch={mismatch[-1]:.3f} "
              f"recall_gain={g_recall[-1]:+.3f} fid_gain(vs win/attn)={g_fid_win[-1]:+.3f}/{g_fid_attn[-1]:+.3f}",
              flush=True)
        common.cleanup_clusters(model)

    np.savez(
        args.out, budget=args.budget, dataset=args.dataset,
        mismatch=np.array(mismatch), gain_recall=np.array(g_recall),
        gain_fid_vs_window=np.array(g_fid_win), gain_fid_vs_attn=np.array(g_fid_attn),
        labels=np.array(labels),
    )
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
