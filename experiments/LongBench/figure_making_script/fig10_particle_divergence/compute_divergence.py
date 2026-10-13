"""Figure 10 (compute stage) - Trajectory disagreement as the confidence signal.

Paper App. F.4. Relates the query-space disagreement of Eq. 10,

    D_m = 1/(M-1) sum_{r!=m} 1/(T|G|) sum_{u,g} [ 1 - cos(q^m_{u,g}, q^r_{u,g}) ],

standardized across the M trajectories (App. C.3), and the entropy channel of the
reliability score R_m (Eq. 3) to each trajectory's oracle utility.

Two questions have to be answered with data, and this script measures both on the
same captured particles (so the comparison is exactly like-for-like):

 (1) IS D_m THE BETTER SIGNAL?  For every particle we record its oracle utility --
     how well that ONE trajectory covers the true full-cache decode queries
     Q_future, and its recall@B against the oracle perturbation score S* -- and
     pair it with each candidate signal (log-prob, entropy, D_m). A particle that
     disagrees with the ensemble should be a worse proxy for Q_future, i.e. D_m
     should correlate NEGATIVELY with utility, and more strongly than entropy does.
     This is the relationship the figure is named after.

 (2) DOES THE WEIGHTING THAT USES IT WIN?  For each sample we re-weight the same M
     particles under uniform / log-prob / LORE-KV /E / LORE-KV /D weights and measure the
     resulting ensemble coverage of Q_future. Because every scheme sees identical
     particles, any difference is attributable to the weights alone -- no extra
     test-time compute is involved.

Saved: divergence_data.npz (read by plot_divergence.py). Run on the GPU cluster:
  python compute_divergence.py --config config/lorekv-dis-unif-topk.conf \
         --dataset single --samples-per-dataset 4 --num-particles 8 --budget 128
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
    ap.add_argument("--config", default=os.path.join(common.CONFIG_DIR, "lorekv-dis-unif-topk.conf"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--benchmark", default="longbench", choices=["longbench", "ruler"])
    ap.add_argument("--dataset", default="single")
    ap.add_argument("--dataset-path", default=None)
    ap.add_argument("--samples-per-dataset", type=int, default=4)
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--num-particles", type=int, default=8,
                    help="M generated for the diagnostic (D_m needs at least 2)")
    ap.add_argument("--decode-steps", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=64000)
    ap.add_argument("--out", default=None, help="explicit .npz path (overrides the auto tagged name)")
    ap.add_argument("--out-dir", default=None, help="dir for the auto-named tagged .npz (default: this folder)")
    args = ap.parse_args()

    conf = common.parse_conf(args.config)
    conf["NUM_PARTICLES"] = str(args.num_particles)
    model_path = args.model or conf.get("MODEL_PATH", "mistralai/Mistral-7B-Instruct-v0.2")
    args.out = common.default_npz_path(
        args.out, args.out_dir, os.path.dirname(os.path.abspath(__file__)), "divergence_data",
        common.build_tag(args.config, model_path, args.budget, args.dataset))
    ckpt = common.Checkpoint(args.out)
    if os.path.exists(args.out) and not os.path.exists(ckpt.path):
        print(f"{args.out} already complete; skipping recompute"); return

    common.apply_lookahead_patch()
    model, tokenizer = common.load_model_and_tokenizer(model_path)
    common.configure_model(model, conf, args.budget)
    # The diagnostic needs a non-degenerate coefficient on every axis even when the
    # run's own config switched one off, so the weighting rules stay comparable.
    gamma_E = model.model.config.particle_gamma if model.model.config.particle_gamma > 1.0 else 1.3
    lambda_d = model.model.config.lambda_d if model.model.config.lambda_d > 0 else 0.3
    grid = common.weighting_grid(gamma_E=gamma_E, lambda_D=lambda_d)

    # per-particle signals + utilities, and per-sample ensemble coverage per weighting.
    # The grid is the Eq. 3 (beta, gamma, lambda_D) plane: "logprob" (1,1,0),
    # "typicality" (1,0,0) as the no-entropy control, and "typ_div" (1,0,lambda_D).
    strat_keys = list(grid)
    part_keys = ["logp_a", "ent_a", "cbar_a", "div_a", "divz_a", "ualign_a", "urecall_a"]
    state, done = ckpt.load_or({**{k: [] for k in part_keys},
                                "strat": {k: [] for k in strat_keys}})
    logp_a = state["logp_a"]; ent_a = state["ent_a"]; cbar_a = state["cbar_a"]
    div_a = state["div_a"]
    divz_a = state["divz_a"]; ualign_a = state["ualign_a"]; urecall_a = state["urecall_a"]
    strat = state["strat"]

    def _flush():
        ckpt.flush({"logp_a": logp_a, "ent_a": ent_a, "cbar_a": cbar_a,
                    "div_a": div_a, "divz_a": divz_a,
                    "ualign_a": ualign_a, "urecall_a": urecall_a, "strat": strat}, done)

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
        if cap["true_queries"] is None:          # OOM-skipped sample
            common.free_sample(model, cap)
            done.add(key); _flush()
            continue

        M = cap["M"]
        logprob = cap["particle_logprob"]        # [M]
        entropy = cap["particle_entropy"]        # [M]
        divergence = cap["particle_divergence"]  # [M]  raw D_m
        divergence_z = cap["particle_divergence_z"]

        # cbar_m = mean log-prob + mean entropy is the centered channel, recoverable
        # from any capture with no re-decoding.
        c_bar = common.centered_score(logprob, entropy)
        # Every weighting under comparison consumes the SAME M particles.
        w = {k: common.reliability_weights_from_capture(
                 c_bar, entropy, divergence_z, beta=b, gamma=g, lambda_d=ld)
             for k, (b, g, ld) in grid.items()}

        u_align = np.zeros(M)
        u_recall = np.zeros(M)
        cov = {k: [] for k in strat_keys}
        n_used = 0
        for li, layer in enumerate(model.model.layers):
            kvc = layer.self_attn.kv_cluster
            tq = cap["true_queries"][li]
            if kvc.full_key_states is None or tq is None:
                continue
            keys, vals = common.get_repeated_kv(kvc)
            batches, flags = common.make_lookahead_batches(kvc, M)
            scores = kvc._project_and_score(keys, vals, batches, flags)      # [window, p0..]
            oracle = common.oracle_scores(kvc, keys, vals, tq).mean(0).cpu().numpy()

            tqh = tq[0]                                                      # [Hq, Tt, D]
            looks = [q[0] if q is not None else None
                     for q in cap["per_layer"][li]["lookahead_queries"]]
            for m in range(M):
                u_align[m] += common.weighted_coverage(tqh, [looks[m]], torch.ones(1))
                u_recall[m] += common.recall_at_b(scores[1 + m].mean(0).cpu().numpy(),
                                                  oracle, args.budget)
            for k in strat_keys:
                cov[k].append(common.weighted_coverage(tqh, looks, w[k]))
            n_used += 1

        if n_used == 0:
            common.free_sample(model, cap)
            done.add(key); _flush()
            continue
        u_align /= n_used; u_recall /= n_used

        logp_a.append(logprob.numpy()); ent_a.append(entropy.numpy())
        cbar_a.append(c_bar.numpy())
        div_a.append(divergence.numpy()); divz_a.append(divergence_z.numpy())
        ualign_a.append(u_align); urecall_a.append(u_recall)
        for k in strat_keys:
            strat[k].append(float(np.mean(cov[k])))
        done.add(key); _flush()
        print(f"[{sample['dataset']}#{sample['sample_idx']}] n={cap['n']} "
              f"D=[{divergence.min():.4f},{divergence.max():.4f}]  "
              f"cov: unif={strat['uniform'][-1]:.4f} typ={strat['typicality'][-1]:.4f} "
              f"ent={strat['entropy'][-1]:.4f} typ_div={strat['typ_div'][-1]:.4f}", flush=True)
        common.free_sample(model, cap)

    if not logp_a:
        print("No usable samples (every sample was skipped); nothing saved.")
        return

    np.savez(
        args.out,
        budget=args.budget, dataset=args.dataset,
        gamma_E=gamma_E, lambda_d=lambda_d,
        grid=np.array([[k, *map(str, v)] for k, v in grid.items()]),
        n_samples=len(logp_a),
        logprob=np.concatenate(logp_a), entropy=np.concatenate(ent_a),
        c_bar=np.concatenate(cbar_a),
        divergence=np.concatenate(div_a), divergence_z=np.concatenate(divz_a),
        util_align=np.concatenate(ualign_a), util_recall=np.concatenate(urecall_a),
        strat_keys=np.array(strat_keys),
        strat_mean=np.array([np.mean(strat[k]) for k in strat_keys]),
        strat_std=np.array([np.std(strat[k]) for k in strat_keys]),
        # paired per-sample coverages: the weighting comparison is only meaningful
        # sample-by-sample (absolute coverage varies far more across prompts than
        # across weightings), so keep the pairing for the win-rate panel.
        strat_per_sample=np.array([strat[k] for k in strat_keys]),
    )
    ckpt.done()
    print(f"Saved {args.out}  ({len(logp_a)} samples)")


if __name__ == "__main__":
    main()
