"""Figure 1 (compute stage) - Future-query alignment.

Validates the paper's central premise: do confidence-weighted MULTIPLE lookahead
trajectories approximate the queries that actually drive decoding better than a
prompt-window reference or a single surrogate trajectory?

For each sample we obtain the TRUE decoding queries Q_true by greedy-decoding the
answer against the FULL (uncompressed) cache, then measure how well several
estimated query sets cover them (cosine coverage, common.weighted_coverage):

  * prompt-window queries              (what SnapKV/AnDPro score with)
  * a single lookahead trajectory      (Lookahead Q-Cache style)
  * M trajectories, uniform weights
  * M trajectories, log-probability weighting
  * M trajectories, LORE-KV log-prob - entropy weighting

Saved: alignment_data.npz (read by plot_alignment.py). No figure is drawn here.

Run on the GPU cluster, e.g.:
  python compute_alignment.py --config config/tlckv-pe-mx-unif-topk.conf \
         --dataset single --samples-per-dataset 3 --budget 128
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(common.CONFIG_DIR, "tlckv-pe-m4-adap-topk.conf"),
                    help="repo .conf whose hyper-parameters (eta, lambda_h, T, ...) drive the run")
    ap.add_argument("--model", default=None, help="override MODEL_PATH from the config")
    ap.add_argument("--benchmark", default="longbench", choices=["longbench", "ruler"])
    ap.add_argument("--dataset", default="single", help="'single' (single-doc QA), 'full', or comma-list")
    ap.add_argument("--dataset-path", default=None, help="override HF path (defaults per benchmark)")
    ap.add_argument("--samples-per-dataset", type=int, default=3)
    ap.add_argument("--budget", type=int, default=128, help="KV budget B (tightest = most diagnostic)")
    ap.add_argument("--m-list", type=int, nargs="+", default=[1, 2, 4, 8],
                    help="particle counts to plot on the x-axis; max is generated once")
    ap.add_argument("--decode-steps", type=int, default=32, help="# true decode queries per sample")
    ap.add_argument("--max-length", type=int, default=64000)
    ap.add_argument("--out", default=None, help="explicit .npz path (overrides the auto tagged name)")
    ap.add_argument("--out-dir", default=None, help="dir for the auto-named tagged .npz (default: this folder)")
    args = ap.parse_args()

    conf = common.parse_conf(args.config)
    m_max = max(args.m_list)
    # force enough particles to be generated; keep everything else from the config
    conf["NUM_PARTICLES"] = str(m_max)
    model_path = args.model or conf.get("MODEL_PATH", "mistralai/Mistral-7B-Instruct-v0.2")
    args.out = common.default_npz_path(
        args.out, args.out_dir, os.path.dirname(os.path.abspath(__file__)), "alignment_data",
        common.build_tag(args.config, model_path, args.budget, args.dataset))

    ckpt = common.Checkpoint(args.out)
    if os.path.exists(args.out) and not os.path.exists(ckpt.path):
        print(f"{args.out} already complete; skipping recompute"); return

    common.apply_lookahead_patch()
    model, tokenizer = common.load_model_and_tokenizer(model_path)
    common.configure_model(model, conf, args.budget)
    eta = model.model.config.particle_eta
    lambda_h = model.model.config.lambda_h

    strategies = ["prompt_window", "single_traj", "uniform", "logprob", "lorekv"]
    # per-strategy: list (over samples) of arrays (over m_list) of mean coverage; resumable.
    state, done = ckpt.load_or({"acc": {s: [] for s in strategies}, "n_samples": 0})
    acc = state["acc"]; n_samples = state["n_samples"]

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
            done.add(key); ckpt.flush({"acc": acc, "n_samples": n_samples}, done)
            continue

        logprob = cap["particle_logprob"]
        entropy = cap["particle_entropy"]
        n_layers = len(cap["per_layer"])

        per_m = {s: [] for s in strategies}   # each -> list over m_list of layer-averaged coverage
        for M in args.m_list:
            lp = logprob[:M]
            en = entropy[:M]
            w_uniform = torch.full((M,), 1.0 / M)
            w_logprob = common.softmax_weights(lp, eta)
            w_lorekv = common.softmax_weights(lp - lambda_h * en, eta)

            layer_cov = {s: [] for s in strategies}
            for li in range(n_layers):
                tq = cap["true_queries"][li]
                if tq is None:
                    continue
                tq = tq[0]                                   # [Hq, Tt, D]
                pl = cap["per_layer"][li]
                win = pl["window_queries"][0]                # [Hq, w, D]
                looks = [q[0] if q is not None else None for q in pl["lookahead_queries"][:M]]

                layer_cov["prompt_window"].append(common.weighted_coverage(tq, [win], torch.ones(1)))
                layer_cov["single_traj"].append(common.weighted_coverage(tq, [looks[0]], torch.ones(1)))
                layer_cov["uniform"].append(common.weighted_coverage(tq, looks, w_uniform))
                layer_cov["logprob"].append(common.weighted_coverage(tq, looks, w_logprob))
                layer_cov["lorekv"].append(common.weighted_coverage(tq, looks, w_lorekv))
            for s in strategies:
                per_m[s].append(float(np.mean(layer_cov[s])) if layer_cov[s] else np.nan)

        for s in strategies:
            acc[s].append(per_m[s])
        n_samples += 1
        done.add(key); ckpt.flush({"acc": acc, "n_samples": n_samples}, done)
        print(f"[{sample['dataset']}#{sample['sample_idx']}] n={cap['n']} "
              f"lorekv@M={args.m_list[-1]}: {per_m['lorekv'][-1]:.4f}", flush=True)
        common.free_sample(model, cap)

    means = {s: np.nanmean(np.array(acc[s]), axis=0) for s in strategies}
    stds = {s: np.nanstd(np.array(acc[s]), axis=0) for s in strategies}

    np.savez(
        args.out,
        m_list=np.array(args.m_list),
        strategies=np.array(strategies),
        n_samples=n_samples,
        eta=eta, lambda_h=lambda_h, budget=args.budget,
        dataset=args.dataset, decode_steps=args.decode_steps,
        **{f"mean_{s}": means[s] for s in strategies},
        **{f"std_{s}": stds[s] for s in strategies},
    )
    ckpt.done()
    print(f"Saved {args.out}  ({n_samples} samples)")


if __name__ == "__main__":
    main()
