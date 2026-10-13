"""Capture the Monte Carlo lookahead trajectories of real prompts (paper Figure 1, App. F.2).

For each prompt the script records:

* the text of each sampled continuation and of the answer from the full cache;
* the post-RoPE queries of the prompt window, of each trajectory and of the true
  decoding steps, with a 2-D PCA of their union;
* the coverage of the true queries by each trajectory, and its weight omega_m;
* the coverage as a function of M, over nested prefixes of the same sample.

The script uses the same patches, configuration and buffers as
``scripts/demo_token_ranking.py``.

Usage (on a GPU node)::

    python scripts/demo_montecarlo_queries.py --benchmark ruler --dataset 4096 \
        --samples-per-dataset 1 --particles 8
    python scripts/plot_montecarlo_demo.py

Output: ``demos/montecarlo/<tag>/sample_<i>.npz`` and ``sample_<i>.json`` (the texts).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments", "LongBench", "figure_making_script"))
sys.path.insert(0, REPO_ROOT)

import common


def unit(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


@torch.no_grad()
def head_coverage(true_q, src):
    """Return the coverage of the true queries by one query source, for each head.

    The coverage is the mean, over the true steps, of the best cosine to a query of
    the source (Eq. 13 for one source).

    Args:
        true_q: Tensor [H, T, D].
        src: Tensor [H, Q, D].

    Returns:
        Tensor [H].
    """
    tq, s = unit(true_q.float()), unit(src.float())
    sim = torch.matmul(tq, s.transpose(1, 2))          # [H, T, Q]
    return sim.max(dim=-1).values.mean(dim=-1)         # [H]


@torch.no_grad()
def capture(model, tokenizer, input_ids, attention_mask, decode_steps, cfg):
    """Run the prefill, the true greedy decode and M sampled trajectories, and keep the token IDs.

    This is a short copy of ``common._collect_and_lookahead_impl`` that also keeps the
    sampled and greedy token IDs.
    """
    from lorekv.monkeypatch.lookahead_runner import _shape_logits
    from lorekv.monkeypatch.scoring import (
        particle_trajectory_divergence, reliability_weights, standardize_disagreement,
        step_stats,
    )

    device = input_ids.device
    n = input_ids.shape[1]
    M, T = cfg.num_particles, cfg.lookahead_len
    temp = cfg.lookahead_temperature
    top_k = getattr(cfg, "lookahead_top_k", None)
    top_p = getattr(cfg, "lookahead_top_p", None)

    cfg.lookahead_phase = "collect"
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    base_cache = out.past_key_values
    last_logits = out.logits[:, -1, :].float()

    # --- the future the model actually produces, from the FULL cache ---------
    cap = common.PostRoPEQueryCapture(model)
    cfg.lookahead_phase = "lookahead"
    true_ids = []
    with cap:
        cap.enabled = True
        tcache = common._clone_dynamic_cache(base_cache)
        tok = last_logits.argmax(dim=-1, keepdim=True)
        pos = n
        for _ in range(decode_steps):
            true_ids.append(int(tok.item()))
            step = model(input_ids=tok, past_key_values=tcache, use_cache=True,
                         cache_position=torch.tensor([pos], device=device),
                         position_ids=torch.tensor([[pos]], device=device))
            tcache = step.past_key_values
            tok = step.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            pos += 1
        cap.enabled = False
    true_queries = cap.stacked()
    for layer in model.model.layers:
        layer.self_attn.kv_cluster.reset_lookahead_buffers()

    # --- the M sampled trajectories -----------------------------------------
    cfg.lookahead_phase = "lookahead"
    particle_ids = []
    pc = torch.zeros(M, T, device=device)
    pH = torch.zeros(M, T, device=device)
    pV = torch.zeros(M, T, device=device)
    for m in range(M):
        cfg.lookahead_particle_idx = m
        cache_m = common._clone_dynamic_cache(base_cache)
        q0 = torch.log_softmax(_shape_logits(last_logits, temp, top_k, top_p), dim=-1).exp()
        tok = torch.multinomial(q0, num_samples=1)
        ids, cs, hs, vs = [], [], [], []
        pos = n
        for _ in range(T):
            ids.append(int(tok.item()))
            step = model(input_ids=tok, past_key_values=cache_m, use_cache=True,
                         cache_position=torch.tensor([pos], device=device),
                         position_ids=torch.tensor([[pos]], device=device))
            logits = step.logits[:, -1, :].float()
            z = _shape_logits(logits, temp, top_k, top_p)
            nxt = torch.multinomial(torch.softmax(z, dim=-1), num_samples=1)
            _, h_u, v_u, c_u = step_stats(logits, nxt.squeeze(-1), temperature=temp,
                                          top_k=top_k, top_p=top_p)
            cs.append(c_u.squeeze()); hs.append(h_u.squeeze()); vs.append(v_u.squeeze())
            cache_m = step.past_key_values
            tok = nxt
            pos += 1
        particle_ids.append(ids)
        pc[m] = torch.stack(cs); pH[m] = torch.stack(hs); pV[m] = torch.stack(vs)

    dz = standardize_disagreement(
        particle_trajectory_divergence(model.model.layers, M, layer_stride=1, device=device))
    weights, _ = reliability_weights(pc, pH, pV, dz, beta=cfg.particle_beta,
                                     gamma=cfg.particle_gamma, lambda_d=cfg.lambda_d,
                                     normalize=bool(getattr(cfg, "normalize_typicality", False)))

    per_layer = []
    for layer in model.model.layers:
        kvc = layer.self_attn.kv_cluster
        look = [torch.cat(kvc._lookahead_buffers[m], dim=2) if kvc._lookahead_buffers[m] else None
                for m in range(M)]
        per_layer.append({"window": kvc.window_queries, "look": look})

    del base_cache
    torch.cuda.empty_cache()
    return dict(n=n, M=M, T=T, true_queries=true_queries, per_layer=per_layer,
                weights=weights.cpu(), particle_ids=particle_ids, true_ids=true_ids,
                entropy=pH.mean(-1).cpu(), c_bar=pc.mean(-1).cpu())


def pca2(X):
    """Compute a 2-D PCA of unit-normalized rows. Return the coordinates and the explained variance."""
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ Vt[:2].T
    var = (S ** 2) / max(1e-12, (S ** 2).sum())
    return coords, float(var[:2].sum())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "config", "lorekv-ent-unif-topk.conf"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--benchmark", default="ruler", choices=["longbench", "ruler"])
    ap.add_argument("--dataset", default="4096")
    ap.add_argument("--dataset-path", default=None)
    ap.add_argument("--samples-per-dataset", type=int, default=1)
    ap.add_argument("--particles", type=int, default=8, help="override NUM_PARTICLES")
    ap.add_argument("--lookahead-len", type=int, default=None, help="override LOOKAHEAD_LEN")
    ap.add_argument("--decode-steps", type=int, default=24)
    ap.add_argument("--max-length", type=int, default=12000)
    ap.add_argument("--layer", type=int, default=-1, help="layer whose queries are projected (-1 = middle)")
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "demos", "montecarlo"))
    args = ap.parse_args()

    conf = common.parse_conf(args.config)
    model_path = args.model or conf.get("MODEL_PATH", "mistralai/Mistral-7B-Instruct-v0.2")
    common.apply_lookahead_patch()
    model, tokenizer = common.load_model_and_tokenizer(model_path)
    common.configure_model(model, conf, 128)
    cfg = model.model.config
    cfg.num_particles = args.particles
    if args.lookahead_len:
        cfg.lookahead_len = args.lookahead_len

    tag = "__".join([common._sanitize(str(model_path).rstrip("/").split("/")[-1]),
                     common._sanitize(args.benchmark), common._sanitize(args.dataset),
                     f"M{cfg.num_particles}", f"T{cfg.lookahead_len}"])
    out_dir = os.path.join(args.out_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[mc] writing to {out_dir}", flush=True)

    for sample in common.load_samples(args.benchmark, tokenizer, args.dataset, model_path,
                                      dataset_path=args.dataset_path,
                                      max_length=args.max_length,
                                      samples_per_dataset=args.samples_per_dataset):
        t0 = time.time()
        capd = capture(model, tokenizer, sample["input_ids"], sample["attention_mask"],
                       args.decode_steps, cfg)
        M, T = capd["M"], capd["T"]
        n_layers = len(model.model.layers)
        li = args.layer if args.layer >= 0 else n_layers // 2
        pl = capd["per_layer"][li]
        true_q = capd["true_queries"][li][0]                      # [H, Tt, D]
        win_q = pl["window"][0]                                   # [H, w, D]
        look_q = [pl["look"][m][0] for m in range(M)]             # each [H, T, D]

        # coverage per head, so the projected head is the one that shows the effect
        cov_win = head_coverage(true_q, win_q)                                  # [H]
        cov_look = head_coverage(true_q, torch.cat(look_q, dim=1))              # [H]
        head = int(torch.argmax(cov_look - cov_win).item())

        # per-particle coverage (whole-model mean and on the projected head)
        cov_particle = np.array([float(head_coverage(true_q, q).mean()) for q in look_q])
        cov_particle_head = np.array([float(head_coverage(true_q, q)[head]) for q in look_q])

        # coverage vs M over nested prefixes of the SAME draw (uniform weights)
        cov_vs_m = []
        for k in range(1, M + 1):
            cov_vs_m.append(float(head_coverage(true_q, torch.cat(look_q[:k], dim=1)).mean()))

        # 2-D projection of the queries of the chosen head
        W = win_q[head].float().cpu().numpy()
        P = np.concatenate([q[head].float().cpu().numpy() for q in look_q], 0)
        F = true_q[head].float().cpu().numpy()
        coords, evr = pca2(np.concatenate([W, P, F], 0))
        nW, nP = len(W), len(P)

        arrays = dict(
            coords=coords.astype(np.float32), evr=evr, n_window=nW, n_particle_total=nP,
            M=M, T=T, layer=li, head=head, n=capd["n"],
            weights=capd["weights"].numpy(), entropy=capd["entropy"].numpy(),
            c_bar=capd["c_bar"].numpy(),
            cov_window=float(cov_win.mean()), cov_lookahead=float(cov_look.mean()),
            cov_window_head=float(cov_win[head]), cov_lookahead_head=float(cov_look[head]),
            cov_particle=cov_particle, cov_particle_head=cov_particle_head,
            cov_vs_m=np.array(cov_vs_m), dataset=sample["dataset"],
            sample_idx=sample["sample_idx"], context_length=sample["context_length"],
        )
        base = f"{common._sanitize(sample['dataset'])}__s{sample['sample_idx']}"
        np.savez_compressed(os.path.join(out_dir, base + ".npz"), **arrays)

        texts = dict(
            dataset=sample["dataset"], sample_idx=sample["sample_idx"],
            prompt_tail=tokenizer.decode(sample["input_ids"][0, -60:], skip_special_tokens=True),
            answers=[str(a) for a in (sample.get("answers") or [])],
            true_continuation=tokenizer.decode(capd["true_ids"], skip_special_tokens=True),
            particles=[{"m": m,
                        "text": tokenizer.decode(capd["particle_ids"][m], skip_special_tokens=True),
                        "omega": float(capd["weights"][m]),
                        "entropy": float(capd["entropy"][m]),
                        "coverage": float(cov_particle[m])}
                       for m in range(M)],
        )
        with open(os.path.join(out_dir, base + ".json"), "w") as fh:
            json.dump(texts, fh, indent=2)

        print(f"[{sample['dataset']}#{sample['sample_idx']}] n={capd['n']} "
              f"{time.time()-t0:.0f}s  layer={li} head={head}  "
              f"coverage window={float(cov_win.mean()):.3f} -> lookahead={float(cov_look.mean()):.3f}",
              flush=True)
        common.free_sample(model, None)
        common.cleanup_clusters(model)

    print(f"[mc] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
