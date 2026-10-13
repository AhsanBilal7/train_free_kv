"""Trajectory entropy and coverage, compute stage (paper App. F.4).

For each trajectory the script records its mean entropy (H_bar in Eq. 3) and the
coverage of the true queries by that trajectory alone. For each sample it records
the coverage of (a) one random trajectory, (b) M trajectories with uniform weights,
(c) log-probability weights and (d) LORE-KV /E weights.

Output: ``entropy_data.npz`` for ``plot_entropy.py``.

Usage (one GPU)::

    python compute_entropy.py --config config/lorekv-ent-unif-topk.conf \\
        --dataset single --samples-per-dataset 4 --budget 128
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
    ap.add_argument("--config", default=os.path.join(common.CONFIG_DIR, "lorekv-ent-adap-topk.conf"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--benchmark", default="longbench", choices=["longbench", "ruler"])
    ap.add_argument("--dataset", default="single")
    ap.add_argument("--dataset-path", default=None)
    ap.add_argument("--samples-per-dataset", type=int, default=4)
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--num-particles", type=int, default=8, help="M generated for the diagnostic")
    ap.add_argument("--decode-steps", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=64000)
    ap.add_argument("--out", default=None, help="explicit .npz path (overrides the auto tagged name)")
    ap.add_argument("--out-dir", default=None, help="dir for the auto-named tagged .npz (default: this folder)")
    args = ap.parse_args()

    conf = common.parse_conf(args.config)
    conf["NUM_PARTICLES"] = str(args.num_particles)
    model_path = args.model or conf.get("MODEL_PATH", "mistralai/Mistral-7B-Instruct-v0.2")
    args.out = common.default_npz_path(
        args.out, args.out_dir, os.path.dirname(os.path.abspath(__file__)), "entropy_data",
        common.build_tag(args.config, model_path, args.budget, args.dataset))
    ckpt = common.Checkpoint(args.out)
    if os.path.exists(args.out) and not os.path.exists(ckpt.path):
        print(f"{args.out} already complete; skipping recompute"); return
    common.apply_lookahead_patch()
    model, tokenizer = common.load_model_and_tokenizer(model_path)
    common.configure_model(model, conf, args.budget)
    gamma_E = model.model.config.particle_gamma if model.model.config.particle_gamma > 1.0 else 1.3

    strat_keys = ["single_mean", "single_min", "single_max", "uniform", "logprob", "lorekv"]
    state, done = ckpt.load_or({"ent_all": [], "logp_all": [], "align_all": [],
                               "strat": {k: [] for k in strat_keys}})
    ent_all = state["ent_all"]; logp_all = state["logp_all"]; align_all = state["align_all"]
    strat = state["strat"]

    def _flush():
        ckpt.flush({"ent_all": ent_all, "logp_all": logp_all,
                    "align_all": align_all, "strat": strat}, done)

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

        M = cap["M"]
        logprob = cap["particle_logprob"]      # [M]
        entropy = cap["particle_entropy"]      # [M]
        n_layers = len(cap["per_layer"])

        # per-particle single-trajectory alignment, averaged over layers
        per_particle_align = np.zeros(M)
        # strategy coverages, averaged over layers
        cov_uniform, cov_logprob, cov_lorekv = [], [], []
        w_uniform = torch.full((M,), 1.0 / M)
        # log-probability only: (beta, gamma, lambda_D) = (1, 1, 0).
        w_logprob = common.reliability_weights_from_capture(
            common.centered_score(logprob, entropy), entropy, beta=1.0, gamma=1.0)
        # LORE-KV /E: (beta, gamma, lambda_D) = (1, gamma_E, 0).
        w_lorekv = common.reliability_weights_from_capture(
            common.centered_score(logprob, entropy), entropy, beta=1.0, gamma=gamma_E)

        for li in range(n_layers):
            tq = cap["true_queries"][li]
            if tq is None:
                continue
            tq = tq[0]
            looks = [q[0] if q is not None else None for q in cap["per_layer"][li]["lookahead_queries"]]
            for m in range(M):
                per_particle_align[m] += common.weighted_coverage(tq, [looks[m]], torch.ones(1))
            cov_uniform.append(common.weighted_coverage(tq, looks, w_uniform))
            cov_logprob.append(common.weighted_coverage(tq, looks, w_logprob))
            cov_lorekv.append(common.weighted_coverage(tq, looks, w_lorekv))
        per_particle_align /= max(1, n_layers)

        ent_all.append(entropy.numpy()); logp_all.append(logprob.numpy())
        align_all.append(per_particle_align)
        strat["single_mean"].append(float(per_particle_align.mean()))
        strat["single_min"].append(float(per_particle_align.min()))
        strat["single_max"].append(float(per_particle_align.max()))
        strat["uniform"].append(float(np.mean(cov_uniform)))
        strat["logprob"].append(float(np.mean(cov_logprob)))
        strat["lorekv"].append(float(np.mean(cov_lorekv)))
        done.add(key); _flush()
        print(f"[{sample['dataset']}#{sample['sample_idx']}] "
              f"single(min/mean/max)={strat['single_min'][-1]:.3f}/{strat['single_mean'][-1]:.3f}/"
              f"{strat['single_max'][-1]:.3f}  lorekv={strat['lorekv'][-1]:.3f}", flush=True)
        common.free_sample(model, cap)

    np.savez(
        args.out,
        budget=args.budget, dataset=args.dataset, gamma_E=gamma_E,
        entropy=np.concatenate(ent_all), logprob=np.concatenate(logp_all),
        align=np.concatenate(align_all),
        strat_keys=np.array(strat_keys),
        strat_mean=np.array([np.mean(strat[k]) for k in strat_keys]),
        strat_std=np.array([np.std(strat[k]) for k in strat_keys]),
    )
    ckpt.done()
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
