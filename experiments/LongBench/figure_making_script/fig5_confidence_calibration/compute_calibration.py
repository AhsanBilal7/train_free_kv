"""Figure 5 (compute stage) - Confidence-weight calibration.

Do the probability-entropy weights actually pick useful trajectories? For every
lookahead particle m we measure its ORACLE utility (how well that one trajectory's
query states match the true full-cache decode path), then relate it to each
confidence signal:

  utility U_m   :  (a) alignment  = cosine coverage of Q_true by particle m's queries
                   (b) recall@B   = top-B overlap of particle m's perturbation score
                                    with the oracle score S* (scored under Q_true)
  signals       :  mean log-prob, mean entropy, trajectory disagreement D_m,
                   combined R_m = logp - lambda_H*H, normalized weight omega_m

Terminology note: this is ordinary categorical NEXT-TOKEN entropy, NOT semantic
entropy (which groups meaning-equivalent generations). Keep them distinct.

Saved: calibration_data.npz. Run on the GPU cluster, e.g.:
  python compute_calibration.py --config config/lorekv-ent-unif-topk.conf \
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
    ap.add_argument("--config", default=os.path.join(common.CONFIG_DIR, "lorekv-ent-adap-topk.conf"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--benchmark", default="longbench", choices=["longbench", "ruler"])
    ap.add_argument("--dataset", default="single")
    ap.add_argument("--dataset-path", default=None)
    ap.add_argument("--samples-per-dataset", type=int, default=4)
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--num-particles", type=int, default=8)
    ap.add_argument("--decode-steps", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=64000)
    ap.add_argument("--out", default=None, help="explicit .npz path (overrides the auto tagged name)")
    ap.add_argument("--out-dir", default=None, help="dir for the auto-named tagged .npz (default: this folder)")
    args = ap.parse_args()

    conf = common.parse_conf(args.config)
    conf["NUM_PARTICLES"] = str(args.num_particles)
    model_path = args.model or conf.get("MODEL_PATH", "mistralai/Mistral-7B-Instruct-v0.2")
    args.out = common.default_npz_path(
        args.out, args.out_dir, os.path.dirname(os.path.abspath(__file__)), "calibration_data",
        common.build_tag(args.config, model_path, args.budget, args.dataset))
    ckpt = common.Checkpoint(args.out)
    if os.path.exists(args.out) and not os.path.exists(ckpt.path):
        print(f"{args.out} already complete; skipping recompute"); return
    common.apply_lookahead_patch()
    model, tokenizer = common.load_model_and_tokenizer(model_path)
    common.configure_model(model, conf, args.budget)
    gamma_E = model.model.config.particle_gamma if model.model.config.particle_gamma > 1.0 else 1.3
    lambda_L = model.model.config.lambda_lookahead

    keys6 = ["logp_a", "ent_a", "cbar_a", "div_a", "divz_a", "comb_a", "w_a",
             "ualign_a", "urecall_a"]
    state, done = ckpt.load_or({k: [] for k in keys6})
    logp_a = state["logp_a"]; ent_a = state["ent_a"]; comb_a = state["comb_a"]
    cbar_a = state["cbar_a"]; div_a = state["div_a"]; divz_a = state["divz_a"]
    w_a = state["w_a"]; ualign_a = state["ualign_a"]; urecall_a = state["urecall_a"]

    def _flush():
        ckpt.flush({"logp_a": logp_a, "ent_a": ent_a, "cbar_a": cbar_a,
                    "div_a": div_a, "divz_a": divz_a, "comb_a": comb_a, "w_a": w_a,
                    "ualign_a": ualign_a, "urecall_a": urecall_a}, done)

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
        logprob = cap["particle_logprob"]; entropy = cap["particle_entropy"]
        divergence = cap["particle_divergence"]; divergence_z = cap["particle_divergence_z"]
        # Eq. 3 at the /E cell in the new coordinates: (beta, gamma, lambda_D) =
        # (1, 1 + lambda_H_old, 0). cbar = logp_mean + H_mean is the centered channel.
        c_bar = common.centered_score(logprob, entropy)
        combined = common.reliability_R(c_bar, entropy, beta=1.0, gamma=gamma_E)
        weights = torch.softmax(combined, dim=0)
        n_layers = len(model.model.layers)

        u_align = np.zeros(M)
        u_recall = np.zeros(M)
        n_used = 0
        for li, layer in enumerate(model.model.layers):
            kvc = layer.self_attn.kv_cluster
            tq = cap["true_queries"][li]
            if kvc.full_key_states is None or tq is None:
                continue
            # NB: named k_rep/v_rep, NOT key/val -- `key` is the checkpoint entry id from
            # the top of this loop, and shadowing it made done.add(key) store a CUDA
            # tensor, so resume never skipped a finished sample.
            k_rep, v_rep = common.get_repeated_kv(kvc)
            batches, flags = common.make_lookahead_batches(kvc, M)
            scores = kvc._project_and_score(k_rep, v_rep, batches, flags)      # [window, p0..]
            oracle = common.oracle_scores(kvc, k_rep, v_rep, tq).mean(0).cpu().numpy()
            tqh = tq[0]
            looks = [q[0] if q is not None else None for q in cap["per_layer"][li]["lookahead_queries"]]
            for m in range(M):
                u_align[m] += common.weighted_coverage(tqh, [looks[m]], torch.ones(1))
                s_m = scores[1 + m].mean(0).cpu().numpy()
                u_recall[m] += common.recall_at_b(s_m, oracle, args.budget)
            n_used += 1
        if n_used == 0:
            common.free_sample(model, cap)
            done.add(key); _flush()
            continue
        u_align /= n_used; u_recall /= n_used

        logp_a.append(logprob.numpy()); ent_a.append(entropy.numpy())
        div_a.append(divergence.numpy()); divz_a.append(divergence_z.numpy())
        cbar_a.append(c_bar.numpy())
        comb_a.append(combined.numpy()); w_a.append(weights.numpy())
        ualign_a.append(u_align); urecall_a.append(u_recall)
        done.add(key); _flush()
        print(f"[{sample['dataset']}#{sample['sample_idx']}] "
              f"corr(comb,recall)~ per-sample computed in plot; "
              f"best/worst recall={u_recall.max():.3f}/{u_recall.min():.3f}", flush=True)
        common.free_sample(model, cap)

    np.savez(
        args.out, budget=args.budget, dataset=args.dataset, gamma_E=gamma_E,
        logprob=np.concatenate(logp_a), entropy=np.concatenate(ent_a),
        c_bar=np.concatenate(cbar_a),
        divergence=np.concatenate(div_a), divergence_z=np.concatenate(divz_a),
        combined=np.concatenate(comb_a), weight=np.concatenate(w_a),
        util_align=np.concatenate(ualign_a), util_recall=np.concatenate(urecall_a),
    )
    ckpt.done()
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
