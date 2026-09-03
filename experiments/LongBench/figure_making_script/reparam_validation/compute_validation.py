"""Reparametrization validation (compute stage).

Captures everything the V/D/S tests of the reparametrized Eq. 3 need, on one pass
over the prompts, and writes validation_data.npz for report_validation.py.

Eq. 3 in the (beta, gamma, lambda_D) coordinates:

    c_u  = log q(y_u | h_u) + H(q(. | h_u)),      E[c_u | h_u] = 0
    R_m  = beta * cbar_m - gamma * Hbar_m - lambda_D * Dhat_m

What is captured per prompt (P prompts x M particles x T steps):

  c, H, V   per-step centered score, entropy and varentropy  [P, M, T]
  logp      per-step log q(y_u | h_u)                        [P, M, T]
  D, Dhat   raw and standardized trajectory disagreement      [P, M]
  utility   per-particle oracle Recall@B against S*           [P, M]  (--utility)

Tests V1, V2, D1, D2 and S4 need only the trajectory means, which are already
recoverable from any older capture as cbar_m = logp_mean + H_mean. Tests V3, V4
and V5 need the per-step sequence and the varentropy, which is what this pass adds.

Run:
  python compute_validation.py --config ../../../../config/lorekv-ent-unif-topk.conf \
         --dataset single --samples-per-dataset 8 --num-particles 8 --budget 128
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(common.CONFIG_DIR, "lorekv-ent-unif-topk.conf"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--benchmark", default="longbench", choices=["longbench", "ruler"])
    ap.add_argument("--dataset", default="single")
    ap.add_argument("--dataset-path", default=None)
    ap.add_argument("--samples-per-dataset", type=int, default=8)
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--num-particles", type=int, default=8)
    ap.add_argument("--decode-steps", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=64000)
    ap.add_argument("--utility", action="store_true",
                    help="also greedy-decode the true future queries and record each "
                         "particle's oracle Recall@B (needed by tests S3 and S4)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    conf = common.parse_conf(args.config)
    conf["NUM_PARTICLES"] = str(args.num_particles)
    model_path = args.model or conf.get("MODEL_PATH", "mistralai/Mistral-7B-Instruct-v0.2")
    args.out = common.default_npz_path(
        args.out, args.out_dir, os.path.dirname(os.path.abspath(__file__)), "validation_data",
        common.build_tag(args.config, model_path, args.budget, args.dataset))

    ckpt = common.Checkpoint(args.out)
    if os.path.exists(args.out) and not os.path.exists(ckpt.path):
        print(f"{args.out} already complete; skipping recompute"); return

    common.apply_lookahead_patch()
    model, tokenizer = common.load_model_and_tokenizer(model_path)
    common.configure_model(model, conf, args.budget)
    cfg = model.model.config
    beta, gamma = cfg.particle_beta, cfg.particle_gamma
    lambda_d = cfg.lambda_d
    T = cfg.lookahead_len

    keys = ["c", "H", "V", "logp", "D", "Dz", "util", "R"]
    state, done = ckpt.load_or({k: [] for k in keys})
    acc = {k: state[k] for k in keys}

    def _flush():
        ckpt.flush(acc, done)

    for sample in common.load_samples(
        args.benchmark, tokenizer, args.dataset, model_path, dataset_path=args.dataset_path,
        max_length=args.max_length, samples_per_dataset=args.samples_per_dataset,
    ):
        key = common.entry_key(sample)
        if key in done:
            continue
        cap = common.collect_and_lookahead(
            model, sample["input_ids"], sample["attention_mask"],
            decode_steps=args.decode_steps if args.utility else 0,
            capture_true=args.utility,
        )
        if not cap["per_layer"]:                       # OOM-skipped sample
            common.free_sample(model, cap)
            done.add(key); _flush()
            continue

        M = cap["M"]
        acc["c"].append(cap["particle_c"].numpy())
        acc["H"].append(cap["particle_H"].numpy())
        acc["V"].append(cap["particle_V"].numpy())
        acc["logp"].append(cap["particle_logp_steps"].numpy())
        acc["D"].append(cap["particle_divergence"].numpy())
        acc["Dz"].append(cap["particle_divergence_z"].numpy())
        acc["R"].append(cap["particle_R"].numpy())

        # S3/S4 need a per-particle oracle utility: Recall@B of the retention that ONE
        # particle's queries induce, against the retention the true future queries
        # would have kept. Averaged over the layers that supplied both.
        if args.utility and cap["true_queries"] is not None:
            u = np.zeros(M); n_used = 0
            for li, layer in enumerate(model.model.layers):
                kvc = layer.self_attn.kv_cluster
                tq = cap["true_queries"][li]
                if kvc.full_key_states is None or tq is None:
                    continue
                k_rep, v_rep = common.get_repeated_kv(kvc)
                batches, flags = common.make_lookahead_batches(kvc, M)
                scores = kvc._project_and_score(k_rep, v_rep, batches, flags)
                oracle = common.oracle_scores(kvc, k_rep, v_rep, tq).mean(0).cpu().numpy()
                for m in range(M):
                    u[m] += common.recall_at_b(scores[1 + m].mean(0).cpu().numpy(),
                                               oracle, args.budget)
                n_used += 1
            acc["util"].append(u / max(n_used, 1))
        else:
            acc["util"].append(np.full(M, np.nan))

        done.add(key); _flush()
        cb = cap["particle_c"].mean(-1)
        print(f"[{sample['dataset']}#{sample['sample_idx']}] n={cap['n']} "
              f"cbar=[{cb.min():.4f},{cb.max():.4f}] Hbar={cap['particle_entropy'].mean():.3f}",
              flush=True)
        common.free_sample(model, cap)

    if not acc["c"]:
        print("No usable samples; nothing saved.")
        return

    np.savez(
        args.out,
        budget=args.budget, dataset=args.dataset, T=T,
        beta=beta, gamma=gamma, lambda_d=lambda_d,
        temperature=cfg.lookahead_temperature,
        top_k=str(getattr(cfg, "lookahead_top_k", None)),
        top_p=str(getattr(cfg, "lookahead_top_p", None)),
        n_prompts=len(acc["c"]),
        c=np.stack(acc["c"]), H=np.stack(acc["H"]), V=np.stack(acc["V"]),
        logp=np.stack(acc["logp"]),
        D=np.stack(acc["D"]), Dz=np.stack(acc["Dz"]),
        R=np.stack(acc["R"]), util=np.stack(acc["util"]),
    )
    ckpt.done()
    print(f"Saved {args.out}  ({len(acc['c'])} prompts)")


if __name__ == "__main__":
    main()
