"""Figure 7 (compute stage) - What kinds of tokens does each method retain?

Every prompt token is labeled with a coarse semantic class, then we measure how
strongly each eviction rule preserves each class (mean fraction of heads that keep
the token, averaged over layers). Three rules computed from the same machinery:

  * attention      - SnapKV-style mean window attention weight
  * andpro_window  - output-projection perturbation under window queries only
                     (the AnDPro-like, prompt-side output-aware baseline)
  * lorekv_multi   - LORE-KV: window + confidence-weighted lookahead perturbation

Token classes (priority order): attention-sink (pos 0), recent (last window),
answer/evidence (matches a gold answer span), number, capitalized entity,
punctuation/boilerplate, other.

Saved: composition_data.npz. Run on the GPU cluster, e.g.:
  python compute_composition.py --config config/lorekv-ent-adap-topk.conf \
         --dataset single --samples-per-dataset 4 --budget 128
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

CLASSES = ["sink", "recent", "answer/evidence", "number", "entity(cap)", "punct/boiler", "other"]


def classify_tokens(ids, tokenizer, answers, window_size):
    n = len(ids)
    labels = ["other"] * n
    ans_mask = [False] * n
    for ans in (answers or []):
        if not isinstance(ans, str) or not ans.strip():
            continue
        for variant in (ans, " " + ans, ans.strip()):
            aids = tokenizer(variant, add_special_tokens=False).input_ids
            if not aids:
                continue
            L = len(aids)
            for i in range(n - L + 1):
                if ids[i:i + L] == aids:
                    for j in range(i, i + L):
                        ans_mask[j] = True
    for i in range(n):
        s = tokenizer.decode([ids[i]]).strip()
        if i == 0:
            labels[i] = "sink"
        elif i >= n - window_size:
            labels[i] = "recent"
        elif ans_mask[i]:
            labels[i] = "answer/evidence"
        elif any(ch.isdigit() for ch in s):
            labels[i] = "number"
        elif s and all(not ch.isalnum() for ch in s):
            labels[i] = "punct/boiler"
        elif s[:1].isupper() and s.isalpha():
            labels[i] = "entity(cap)"
        else:
            labels[i] = "other"
    return labels


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
    ap.add_argument("--max-length", type=int, default=64000)
    ap.add_argument("--out", default=None, help="explicit .npz path (overrides the auto tagged name)")
    ap.add_argument("--out-dir", default=None, help="dir for the auto-named tagged .npz (default: this folder)")
    args = ap.parse_args()

    conf = common.parse_conf(args.config)
    model_path = args.model or conf.get("MODEL_PATH", "mistralai/Mistral-7B-Instruct-v0.2")
    args.out = common.default_npz_path(
        args.out, args.out_dir, os.path.dirname(os.path.abspath(__file__)), "composition_data",
        common.build_tag(args.config, model_path, args.budget, args.dataset))
    ckpt = common.Checkpoint(args.out)
    if os.path.exists(args.out) and not os.path.exists(ckpt.path):
        print(f"{args.out} already complete; skipping recompute"); return
    common.apply_lookahead_patch()
    model, tokenizer = common.load_model_and_tokenizer(model_path)
    common.configure_model(model, conf, args.budget)
    lambda_L = model.model.config.lambda_lookahead
    window_size = model.model.config.window_size

    methods = ["attention", "andpro_window", "lorekv_multi"]
    state, done = ckpt.load_or({"ret_sum": {m: {c: 0.0 for c in CLASSES} for m in methods},
                               "cls_count": {c: 0 for c in CLASSES}, "total_tokens": 0})
    ret_sum = state["ret_sum"]; cls_count = state["cls_count"]; total_tokens = state["total_tokens"]

    def _flush():
        ckpt.flush({"ret_sum": ret_sum, "cls_count": cls_count, "total_tokens": total_tokens}, done)

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
        weights = cap["particle_weights"]; M = cap["M"]
        ids = sample["input_ids"][0].tolist()
        labels = classify_tokens(ids, tokenizer, sample["answers"], window_size)
        n = len(ids)

        ret_frac = {m: np.zeros(n) for m in methods}
        n_layers_used = 0
        for layer in model.model.layers:
            kvc = layer.self_attn.kv_cluster
            if kvc.full_key_states is None:
                continue
            # k_rep/v_rep, NOT key/val: `key` is this sample's checkpoint entry id, and
            # shadowing it made the later done.add(key) store a CUDA tensor, so resume
            # never skipped a finished sample.
            k_rep, v_rep = common.get_repeated_kv(kvc)
            nn = k_rep.shape[2]
            if nn != n:
                # defensive: the analysis uses the prompt-side tokens the cache actually holds
                n = nn
            batches, flags = common.make_lookahead_batches(kvc, M)
            scores = kvc._project_and_score(k_rep, v_rep, batches, flags)
            s_win, _, s_multi = common.combine_lookahead_scores(scores, weights, lambda_L, M)
            qa = kvc.window_queries[0].float()
            aw = torch.matmul(qa, k_rep[0].transpose(1, 2).float()) / math.sqrt(k_rep.shape[-1])
            s_attn = F.softmax(aw, dim=-1).mean(1)
            method_score = {"attention": s_attn, "andpro_window": s_win, "lorekv_multi": s_multi}
            for m in methods:
                mask = common.head_topb_mask(method_score[m], nn, args.budget, window_size)
                ret_frac[m] += mask.float().mean(0).cpu().numpy()[:len(ret_frac[m])]
            n_layers_used += 1
        if n_layers_used == 0:
            common.free_sample(model, cap)
            done.add(key); _flush()
            continue
        for m in methods:
            ret_frac[m] /= n_layers_used

        for i in range(min(n, len(labels))):
            c = labels[i]
            cls_count[c] += 1
            for m in methods:
                ret_sum[m][c] += float(ret_frac[m][i])
        total_tokens += min(n, len(labels))
        done.add(key); _flush()
        print(f"[{sample['dataset']}#{sample['sample_idx']}] n={n} labeled", flush=True)
        common.free_sample(model, cap)

    retention_rate = {m: np.array([ret_sum[m][c] / cls_count[c] if cls_count[c] else np.nan for c in CLASSES])
                      for m in methods}
    enrichment = {}
    for m in methods:
        tot_ret = sum(ret_sum[m].values())
        enr = []
        for c in CLASSES:
            p_ret = (ret_sum[m][c] / tot_ret) if tot_ret > 0 else 0.0
            p_prompt = (cls_count[c] / total_tokens) if total_tokens else 0.0
            enr.append(p_ret / p_prompt if p_prompt > 0 else np.nan)
        enrichment[m] = np.array(enr)

    save = dict(budget=args.budget, dataset=args.dataset, classes=np.array(CLASSES),
                methods=np.array(methods),
                class_count=np.array([cls_count[c] for c in CLASSES]), total_tokens=total_tokens)
    for m in methods:
        save[f"retention_{m}"] = retention_rate[m]
        save[f"enrichment_{m}"] = enrichment[m]
    np.savez(args.out, **save)
    ckpt.done()
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
