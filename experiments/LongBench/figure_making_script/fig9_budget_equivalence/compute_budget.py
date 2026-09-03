"""Figure 9 (compute stage) - Budget-equivalence / memory-saving curves.

Instead of accuracy, we use full-cache behavioral fidelity F(B): the decode-time
attention-output cosine to the FULL cache achieved at budget B. We sweep B for each
method (attention / AnDPro-window / LORE-KV) and save the F(B) curves; the plot
stage inverts them to get the budget a baseline needs to match LORE-KV's fidelity
and the resulting memory saving.

Saved: budget_data.npz. Run on the GPU cluster, e.g.:
  python compute_budget.py --config config/lorekv-ent-adap-topk.conf \
         --dataset single --samples-per-dataset 3 --budgets 64 128 256 512 1024
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
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(common.CONFIG_DIR, "lorekv-ent-adap-topk.conf"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--benchmark", default="longbench", choices=["longbench", "ruler"])
    ap.add_argument("--dataset", default="single")
    ap.add_argument("--dataset-path", default=None)
    ap.add_argument("--samples-per-dataset", type=int, default=3)
    ap.add_argument("--budgets", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    ap.add_argument("--decode-steps", type=int, default=24)
    ap.add_argument("--score-budget", type=int, default=512,
                    help="budget used to CONFIGURE the model (scores don't depend on it here)")
    ap.add_argument("--max-length", type=int, default=64000)
    ap.add_argument("--out", default=None, help="explicit .npz path (overrides the auto tagged name)")
    ap.add_argument("--out-dir", default=None, help="dir for the auto-named tagged .npz (default: this folder)")
    args = ap.parse_args()

    conf = common.parse_conf(args.config)
    model_path = args.model or conf.get("MODEL_PATH", "mistralai/Mistral-7B-Instruct-v0.2")
    args.out = common.default_npz_path(
        args.out, args.out_dir, os.path.dirname(os.path.abspath(__file__)), "budget_data",
        common.build_tag(args.config, model_path, "sweep", args.dataset))
    ckpt = common.Checkpoint(args.out)
    if os.path.exists(args.out) and not os.path.exists(ckpt.path):
        print(f"{args.out} already complete; skipping recompute"); return
    common.apply_lookahead_patch()
    model, tokenizer = common.load_model_and_tokenizer(model_path)
    common.configure_model(model, conf, args.score_budget)
    lambda_L = model.model.config.lambda_lookahead
    window_size = model.model.config.window_size

    methods = ["attention", "andpro_window", "lorekv_multi"]
    budgets = list(args.budgets)
    state, done = ckpt.load_or({"F_sum": {m: np.zeros(len(budgets)) for m in methods},
                               "F_cnt": {m: np.zeros(len(budgets)) for m in methods}})
    F_sum = state["F_sum"]; F_cnt = state["F_cnt"]

    def _flush():
        ckpt.flush({"F_sum": F_sum, "F_cnt": F_cnt}, done)

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
            # k_rep/v_rep, NOT key/val: `key` is this sample's checkpoint entry id, and
            # shadowing it made the later done.add(key) store a CUDA tensor, so resume
            # never skipped a finished sample.
            k_rep, v_rep = common.get_repeated_kv(kvc)
            n = k_rep.shape[2]
            batches, flags = common.make_lookahead_batches(kvc, M)
            scores = kvc._project_and_score(k_rep, v_rep, batches, flags)
            s_win, _, s_multi = common.combine_lookahead_scores(scores, weights, lambda_L, M)
            qa = kvc.window_queries[0].float()
            aw = torch.matmul(qa, k_rep[0].transpose(1, 2).float()) / math.sqrt(k_rep.shape[-1])
            s_attn = F.softmax(aw, dim=-1).mean(1)
            method_score = {"attention": s_attn, "andpro_window": s_win, "lorekv_multi": s_multi}

            q = tq[0].float(); kf = k_rep[0].float(); vf = v_rep[0].float()
            o_full = common.attn_outputs(q, kf, vf)
            for bi, B in enumerate(budgets):
                for m in methods:
                    mask = common.head_topb_mask(method_score[m], n, B, window_size)
                    o_ev = common.attn_outputs(q, kf, vf, mask)
                    F_sum[m][bi] += float(F.cosine_similarity(o_ev, o_full, dim=-1).mean().item())
                    F_cnt[m][bi] += 1
        done.add(key); _flush()
        print(f"[{sample['dataset']}#{sample['sample_idx']}] n={cap['n']} fidelity swept", flush=True)
        common.free_sample(model, cap)

    save = dict(budgets=np.array(budgets), dataset=args.dataset, methods=np.array(methods))
    for m in methods:
        save[f"fidelity_{m}"] = F_sum[m] / np.maximum(F_cnt[m], 1)
    np.savez(args.out, **save)
    ckpt.done()
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
