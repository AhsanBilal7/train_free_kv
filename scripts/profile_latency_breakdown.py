#!/usr/bin/env python
r"""Stage-wise latency of one request: TTFT / lookahead / scoring / decoding (Figure 7a).

Times the four stages of every sample for the first ``--num_samples`` examples
of LongBench, RULER and NIAH and averages them per (method, budget). It produces
the stage breakdown of Figure 7(a); end-to-end per-sample latency over full
benchmarks (Table 3) comes from ``scripts/report_length_time.py`` instead.

Phase definitions
------------------
andpro / critical (plain `model.generate`-style greedy decode, but driven by a
hand-rolled loop here so the phases are separable):
    ttft      -- the single prefill forward pass, MINUS the time spent inside
                 the method's own update_kv (eviction/scoring) call. This is
                 the "pure compute" prefill time, comparable to lorekv's
                 uncompressed collect-phase prefill below.
    lookahead -- always 0 (no lookahead phase in these methods).
    scoring   -- wall time inside kv_cluster.update_kv during that same
                 prefill forward (measured via a monkeypatched wrapper around
                 AnDProAdaptiveSnapKVCluster.update_kv / AdaptiveSnapKVCluster.
                 update_kv; not a source change, just an in-process wrap).
    decoding  -- every subsequent per-token forward call.

lorekv (causal_lookahead, Algorithm 1's 4 phases -- see
lorekv/monkeypatch/lookahead_runner.py's docstring):
    ttft      -- Phase 1, the uncompressed "collect" prefill.
    lookahead -- Phase 2, the M x T sampled-continuation rollout.
    scoring   -- particle-weight computation + Phase 3's per-layer compress()
                 (Eqs. 3-8): score, allocate, select, evict.
    decoding  -- Phase 4, the real answer's per-token forward calls.

Both families report `ttft` as the non-eviction prefill compute, so it is the
one column directly comparable across methods; `scoring` and `lookahead` are
each method's own added cost on top of it, and `total = ttft + lookahead +
scoring + decoding` is the latency plotted against quality in Figure 7(b).

Timing hooks are pure in-process monkeypatches (see _install_hooks): no repo
file is modified. Each (method, budget) cell loads a fresh model, as a real run
does, since the hijacks are class-level monkeypatches and a stale kv_cluster
from a previous mode must not be reused across models.

Usage (from experiments/LongBench, so config/dataset2*.json resolve)::

    python ../../scripts/profile_latency_breakdown.py \
        --budgets 128 256 512 1024 --num_samples 10 --out latency_breakdown.json
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
LB_DIR = os.path.join(REPO_ROOT, "experiments", "LongBench")
RULER_DIR = os.path.join(REPO_ROOT, "experiments", "RULER")
NIAH_DIR = os.path.join(REPO_ROOT, "experiments", "NIAH")


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # so the module's own `import X` resolve against itself if needed
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Timing hooks -- in-process monkeypatches, no repo file is touched.
# ---------------------------------------------------------------------------
class PhaseClock:
    """Records phase-boundary timestamps for one sample, then reduces to durations."""

    def __init__(self):
        self.marks = {}
        self.scoring_accum = 0.0

    def mark(self, name):
        torch.cuda.synchronize()
        self.marks[name] = time.perf_counter()

    def add_scoring(self, dt):
        self.scoring_accum += dt

    def reset(self):
        self.marks.clear()
        self.scoring_accum = 0.0


def _install_lookahead_hooks(clock):
    """Wrap lookahead_runner._set_phase / reliability_weights for phase marks.

    Sequence inside generate_with_lookahead: _set_phase("collect") ->
    [prefill] -> _set_phase("lookahead") -> [M x T rollout] ->
    reliability_weights(...) -> [compress() x layers] -> _set_phase("final") ->
    [decode loop] -> return.
    """
    lr = sys.modules.get("lorekv.monkeypatch.lookahead_runner")
    if lr is None:
        import lorekv.monkeypatch.lookahead_runner as lr
    orig_set_phase = lr._set_phase
    orig_reliab = lr.reliability_weights

    def wrapped_set_phase(model, phase):
        clock.mark(phase)
        return orig_set_phase(model, phase)

    def wrapped_reliab(*a, **kw):
        clock.mark("scoring")
        return orig_reliab(*a, **kw)

    lr._set_phase = wrapped_set_phase
    lr.reliability_weights = wrapped_reliab
    return lr, orig_set_phase, orig_reliab


def _uninstall_lookahead_hooks(lr, orig_set_phase, orig_reliab):
    lr._set_phase = orig_set_phase
    lr.reliability_weights = orig_reliab


def _install_scoring_hook(cluster_cls, clock):
    orig = cluster_cls.update_kv

    def wrapped(self, *a, **kw):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = orig(self, *a, **kw)
        torch.cuda.synchronize()
        clock.add_scoring(time.perf_counter() - t0)
        return r

    cluster_cls.update_kv = wrapped
    return orig


def _uninstall_scoring_hook(cluster_cls, orig):
    cluster_cls.update_kv = orig


# ---------------------------------------------------------------------------
# Greedy generation loops that expose the phase boundaries.
# ---------------------------------------------------------------------------
def run_baseline(model, input_ids, attention_mask, max_gen, eos_ids, clock, device):
    """andpro / critical: hand-rolled greedy decode (matches pred.py's
    model.generate(do_sample=False, num_beams=1) token-for-token), instrumented
    for ttft/scoring/decoding via the update_kv hook installed by the caller."""
    clock.reset()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    ttft_total = t1 - t0
    scoring = clock.scoring_accum
    ttft = max(0.0, ttft_total - scoring)

    past = out.past_key_values
    logits = out.logits[:, -1, :]
    next_token = logits.argmax(dim=-1, keepdim=True)
    eos_set = set(eos_ids)
    generated = []
    pos = input_ids.shape[1]
    if int(next_token.item()) not in eos_set:
        generated.append(next_token)
        for _ in range(max_gen - 1):
            cache_position = torch.tensor([pos], device=device)
            step_out = model(input_ids=next_token, past_key_values=past, use_cache=True,
                              cache_position=cache_position, position_ids=cache_position.unsqueeze(0))
            past = step_out.past_key_values
            next_token = step_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if int(next_token.item()) in eos_set:
                break
            generated.append(next_token)
            pos += 1
    torch.cuda.synchronize(); t2 = time.perf_counter()
    decoding = t2 - t1
    return dict(ttft=ttft, lookahead=0.0, scoring=scoring, decoding=decoding,
                n_tokens=len(generated))


def run_lorekv(model, tokenizer, input_ids, attention_mask, max_gen, eos_ids, clock):
    from lorekv.monkeypatch.lookahead_runner import generate_with_lookahead
    clock.reset()
    answer_ids = generate_with_lookahead(model, tokenizer, input_ids, attention_mask, max_gen, eos_ids)
    torch.cuda.synchronize(); t_end = time.perf_counter()
    m = clock.marks
    ttft = m["lookahead"] - m["collect"]
    lookahead = m["scoring"] - m["lookahead"]
    scoring = m["final"] - m["scoring"]
    decoding = t_end - m["final"]
    return dict(ttft=ttft, lookahead=lookahead, scoring=scoring, decoding=decoding,
                n_tokens=int(answer_ids.shape[0]))


# ---------------------------------------------------------------------------
# Per-benchmark sample builders. Each yields (input_ids, attention_mask, max_gen).
# ---------------------------------------------------------------------------
def longbench_samples(lb, tokenizer, model_name, num_samples, task, max_length, device):
    dataset2prompt = json.load(open(os.path.join(LB_DIR, "config", "dataset2prompt.json")))
    dataset2maxlen = json.load(open(os.path.join(LB_DIR, "config", "dataset2maxlen.json")))
    # load_longbench_split (not a bare load_dataset call) -- THUDM/LongBench is a
    # loading-script dataset that datasets>=3 can't build; this helper prefers the
    # extracted-jsonl / hashed-offline-cache path pred.py itself uses. See its
    # docstring in experiments/LongBench/pred.py.
    data = lb.load_longbench_split("THUDM/LongBench", task)
    prompt_format = dataset2prompt[task]
    max_gen = dataset2maxlen[task]
    out = []
    for json_obj in list(data)[:num_samples]:
        prompt = prompt_format.format(**json_obj)
        tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized) > max_length:
            half = int(max_length / 2)
            prompt = (tokenizer.decode(tokenized[:half], skip_special_tokens=True)
                      + tokenizer.decode(tokenized[-half:], skip_special_tokens=True))
        if task not in ("trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"):
            prompt = lb.build_chat(tokenizer, prompt, model_name)
        inp = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        eos_ids = [tokenizer.eos_token_id]
        if task == "samsum":
            eos_ids.append(tokenizer.encode("\n", add_special_tokens=False)[-1])
        out.append((inp.input_ids, inp.attention_mask, max_gen, eos_ids))
    return out


def ruler_samples(tokenizer, num_samples, ruler_dataset, ruler_length, max_length, device):
    ruler = _load_module("ruler_pred", os.path.join(RULER_DIR, "pred.py"))
    from datasets import load_dataset
    data = load_dataset(ruler_dataset, str(ruler_length), split="test")
    out = []
    for json_obj in list(data)[:num_samples]:
        input_ids, attention_mask = ruler.build_ruler_input(
            tokenizer, json_obj["context"], json_obj["question"],
            json_obj["answer_prefix"], max_length, device)
        max_gen = int(json_obj["max_new_tokens"])
        out.append((input_ids, attention_mask, max_gen, [tokenizer.eos_token_id]))
    return out


def niah_samples(tokenizer, num_samples, target_length, max_length, device):
    niah = _load_module("niah_pred", os.path.join(NIAH_DIR, "pred.py"))
    haystack_dir = os.path.join(NIAH_DIR, "PaulGrahamEssays")
    essay_texts = niah.read_haystack_texts(haystack_dir)
    essay_token_lists = [tokenizer.encode(t) for t in essay_texts]
    period_ids = niah.build_period_token_ids(tokenizer)
    needle_texts = [niah.DEFAULT_NEEDLE]
    needle_token_lists = [tokenizer.encode(needle_texts[0])]
    depths = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90][:num_samples]
    out = []
    for depth in depths:
        context_tokens = niah.build_context_tokens(essay_token_lists, target_length)
        final_tokens, _idx, _actual = niah.insert_needle_single(
            context_tokens, needle_token_lists, depth, period_ids)
        context_text = tokenizer.decode(final_tokens)
        input_ids, attention_mask = niah.build_niah_input(
            tokenizer, context_text, niah.DEFAULT_QUESTION, niah.DEFAULT_RETRIEVAL_GUIDANCE,
            max_length, device, on_overflow="truncate")
        out.append((input_ids, attention_mask, 64, [tokenizer.eos_token_id]))
    return out


# ---------------------------------------------------------------------------
# Model setup per (method, budget) cell: the same lorekv.harness calls as pred.py.
# ---------------------------------------------------------------------------
def build_model(lb, method, budget, model_path, lorekv_m, lorekv_t):
    from lorekv.harness import (add_method_args, configure_model, install_method,
                                load_model_and_tokenizer)
    mode = {"andpro": "andpro", "critical": "critical", "lorekv": "causal_lookahead"}[method]
    install_method(mode)
    model, tokenizer = load_model_and_tokenizer(model_path)
    # LORE-KV runs its /E core configuration (config/lorekv-ent-unif-topk.conf),
    # which is the harness default; only M and T are profiled here.
    args = add_method_args(argparse.ArgumentParser()).parse_args(
        ["--mode", mode, "--budget", str(budget),
         "--num_particles", str(lorekv_m), "--lookahead_len", str(lorekv_t)])
    configure_model(model, args)
    return model, tokenizer


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--methods", nargs="+", default=["andpro", "critical", "lorekv"])
    ap.add_argument("--budgets", nargs="+", type=int, default=[128, 256, 512, 1024])
    ap.add_argument("--num_samples", type=int, default=10)
    ap.add_argument("--lorekv_m", type=int, default=4)
    ap.add_argument("--lorekv_t", type=int, default=4)
    ap.add_argument("--datasets", nargs="+", default=["longbench", "ruler", "niah"])
    ap.add_argument("--longbench_task", default="qasper")
    ap.add_argument("--ruler_dataset", default="simonjegou/ruler")
    ap.add_argument("--ruler_length", default="4096")
    ap.add_argument("--niah_target_length", type=int, default=4096)
    ap.add_argument("--max_length", type=int, default=64000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lb = _load_module("longbench_pred", os.path.join(LB_DIR, "pred.py"))
    from models_ext import register_extra_architectures  # noqa: F401 (harmless no-op for Mistral)

    from andpro_snapkv.monkeypatch.snapkv_utils import AnDProAdaptiveSnapKVCluster
    from adaptive_snapkv.monkeypatch.snapkv_utils import AdaptiveSnapKVCluster

    device = "cuda:0"
    clock = PhaseClock()
    model_name = os.path.basename(args.model_path)

    results = {}  # {method: {budget: {dataset: {phase: mean_seconds, n_tokens: mean}}}}
    for method in args.methods:
        results[method] = {}
        for budget in args.budgets:
            print(f"=== method={method} budget={budget} ===", flush=True)
            model, tokenizer = build_model(lb, method, budget, args.model_path,
                                            args.lorekv_m, args.lorekv_t)

            hooks = []
            if method == "lorekv":
                hooks.append(("lookahead", _install_lookahead_hooks(clock)))
            elif method == "andpro":
                orig = _install_scoring_hook(AnDProAdaptiveSnapKVCluster, clock)
                hooks.append(("andpro_scoring", (AnDProAdaptiveSnapKVCluster, orig)))
            elif method == "critical":
                orig = _install_scoring_hook(AdaptiveSnapKVCluster, clock)
                hooks.append(("critical_scoring", (AdaptiveSnapKVCluster, orig)))

            cell = {}
            for ds in args.datasets:
                print(f"  dataset={ds}", flush=True)
                if ds == "longbench":
                    samples = longbench_samples(lb, tokenizer, model_name, args.num_samples,
                                                 args.longbench_task, args.max_length, device)
                elif ds == "ruler":
                    samples = ruler_samples(tokenizer, args.num_samples, args.ruler_dataset,
                                             args.ruler_length, args.max_length, device)
                elif ds == "niah":
                    samples = niah_samples(tokenizer, args.num_samples,
                                            args.niah_target_length, args.max_length, device)
                else:
                    raise ValueError(ds)

                per_sample = []
                for i, (input_ids, attention_mask, max_gen, eos_ids) in enumerate(samples):
                    with torch.no_grad():
                        if method == "lorekv":
                            r = run_lorekv(model, tokenizer, input_ids, attention_mask,
                                           max_gen, eos_ids, clock)
                        else:
                            r = run_baseline(model, input_ids, attention_mask, max_gen,
                                              eos_ids, clock, device)
                    r["prompt_tokens"] = int(input_ids.shape[1])
                    per_sample.append(r)
                    print(f"    [{i+1}/{len(samples)}] ttft={r['ttft']:.3f}s "
                          f"lookahead={r['lookahead']:.3f}s scoring={r['scoring']:.3f}s "
                          f"decoding={r['decoding']:.3f}s n_tok={r['n_tokens']} "
                          f"prompt_tok={r['prompt_tokens']}", flush=True)
                    torch.cuda.empty_cache()

                n = len(per_sample)
                avg = {k: sum(s[k] for s in per_sample) / n
                       for k in ("ttft", "lookahead", "scoring", "decoding", "n_tokens",
                                 "prompt_tokens")}
                avg["total"] = avg["ttft"] + avg["lookahead"] + avg["scoring"] + avg["decoding"]
                avg["n_samples"] = n
                avg["per_sample"] = per_sample
                cell[ds] = avg

            for _name, hook in hooks:
                if _name == "lookahead":
                    _uninstall_lookahead_hooks(*hook)
                else:
                    _uninstall_scoring_hook(*hook)

            results[method][budget] = cell

            del model, tokenizer
            torch.cuda.empty_cache()

            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  -> checkpointed to {args.out}", flush=True)

    print(f"Done. Full results in {args.out}")


if __name__ == "__main__":
    main()
