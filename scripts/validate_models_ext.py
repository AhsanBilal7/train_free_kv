#!/usr/bin/env python
"""Compare the models_ext implementation with the upstream transformers implementation.

models_ext implements Ministral3 as a transformers 4.44 Llama subclass, because the
eviction patches do not work with the transformers release that contains the
architecture. This script runs the same checkpoint on the same tokens with both
implementations and compares the logits.

The two implementations need different transformers releases, so the script runs
in two processes::

    # 1. Reference, in the transformers 5.x environment
    $TF5_PY scripts/validate_models_ext.py --impl upstream \\
        --model models/Ministral-3-8B-Instruct-2512 \\
        --ref-model mistralai/Ministral-3-8B-Instruct-2512 --out val/upstream.npz

    # 2. This implementation, in the main environment
    python scripts/validate_models_ext.py --impl vendored \\
        --model models/Ministral-3-8B-Instruct-2512 --out val/vendored.npz

    # 3. Comparison (either environment)
    python scripts/validate_models_ext.py --compare val/upstream.npz val/vendored.npz

Both runs use SDPA and bfloat16, so only the model code is different.

The upstream side reads the published Mistral3 checkpoint (``--ref-model``). The
vendored side reads the text-only checkpoint of ``scripts/convert_ministral3.py``.
The test thus also checks the conversion.
"""

import argparse
import os

import numpy as np
import torch

# Fixed, deterministic probe text. Long enough to exercise several hundred
# positions (so RoPE differences show up) and mixed enough to hit a broad slice
# of the vocabulary. The comparison is on logits, so the content is arbitrary --
# only that both sides see identical token ids matters.
PROBE = (
    "The following passage is a technical description of a caching mechanism.\n\n"
    "In autoregressive transformer inference the key and value tensors produced during "
    "prefill are retained so that each decoding step attends over the whole prefix "
    "without recomputation. The cache grows linearly in the prompt length and in the "
    "number of layers, which makes it the dominant memory cost for long contexts. "
    "Eviction methods choose a subset of positions to keep, trading a small amount of "
    "attention fidelity for a large reduction in memory. The difficulty is that the "
    "queries which will eventually read the cache are not available at the time the "
    "eviction decision is made.\n\n"
    "Question: what makes the eviction decision hard, and what would make it easier?\n"
    "Answer: The decision is hard because"
) * 3


def _load_tokens(model_path, n_tokens, device):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    ids = tok(PROBE, return_tensors="pt").input_ids[:, :n_tokens]
    return ids.to(device)


def run(impl, model_path, ref_model_path, n_tokens, out_path, device, dtype, attn):
    if impl == "vendored":
        from models_ext import register_extra_architectures
        register_extra_architectures()
        load_path = model_path
    else:
        load_path = ref_model_path or model_path

    from transformers import AutoModelForCausalLM

    input_ids = _load_tokens(load_path, n_tokens, device)
    # Surface a partial weight load loudly: a renamed or missing tensor otherwise
    # shows up only as a quality drop much later.
    import transformers
    transformers.logging.set_verbosity_warning()
    kw = dict(torch_dtype=getattr(torch, dtype), device_map=device,
              attn_implementation=attn, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(load_path, **kw).eval()
    except ValueError:
        # The published Ministral3 checkpoint is a Mistral3ForConditionalGeneration,
        # which upstream registers for image-text-to-text, not causal LM. Its forward
        # with input_ids alone runs exactly the text tower and returns lm_head logits,
        # which is what the comparison needs.
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(load_path, **kw).eval()
        print(f"  (loaded via AutoModelForImageTextToText: {type(model).__name__})")

    # A Mistral3 wrapper exposes the text tower under .model.language_model /
    # .language_model; take the causal-LM head's logits either way.
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits.float().cpu().numpy()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez(out_path, logits=logits, input_ids=input_ids.cpu().numpy(),
             impl=impl, model=load_path, dtype=dtype, attn=attn)
    print(f"[{impl}] {load_path}")
    print(f"  logits {logits.shape}  argmax(last)={int(logits[0, -1].argmax())}")
    print(f"  wrote {out_path}")


def compare(path_a, path_b, atol):
    a, b = np.load(path_a, allow_pickle=True), np.load(path_b, allow_pickle=True)
    ids_a, ids_b = a["input_ids"], b["input_ids"]
    if not np.array_equal(ids_a, ids_b):
        raise SystemExit(
            "the two runs saw DIFFERENT token ids, so the logits are not comparable.\n"
            "  (a tokenizer mismatch between the two checkpoints, not a modelling bug)"
        )
    la, lb = a["logits"][0], b["logits"][0]
    diff = np.abs(la - lb)
    # Top-1 agreement is the number that actually decides whether greedy decoding
    # diverges; max|diff| in bf16 is dominated by the largest logits and is noisy.
    top1 = (la.argmax(-1) == lb.argmax(-1)).mean()
    denom = np.linalg.norm(la, axis=-1) * np.linalg.norm(lb, axis=-1)
    cos = ((la * lb).sum(-1) / np.maximum(denom, 1e-12)).mean()
    # Rank correlation of the top slice: catches a systematic reordering that
    # cosine over 150k logits would wash out.
    k = 50
    topk_a = set(np.argsort(-la[-1])[:k].tolist())
    topk_b = set(np.argsort(-lb[-1])[:k].tolist())

    print(f"a = {path_a}  ({a['impl']}, {a['model']})")
    print(f"b = {path_b}  ({b['impl']}, {b['model']})")
    print(f"  positions          {la.shape[0]}, vocab {la.shape[1]}")
    print(f"  max |a-b|          {diff.max():.5f}")
    print(f"  mean |a-b|         {diff.mean():.5f}")
    print(f"  mean cosine        {cos:.8f}")
    print(f"  top-1 agreement    {top1:.4%}  ({int(top1 * la.shape[0])}/{la.shape[0]} positions)")
    print(f"  top-{k} overlap (last pos)  {len(topk_a & topk_b)}/{k}")
    ok = top1 >= 0.99 and cos >= 0.9999 and diff.max() <= atol
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["vendored", "upstream"])
    ap.add_argument("--model", help="checkpoint this implementation should load")
    ap.add_argument("--ref-model", help="checkpoint for the upstream side, if it differs")
    ap.add_argument("--n-tokens", type=int, default=512)
    ap.add_argument("--out")
    ap.add_argument("--device", default="cuda:0")
    # float32 removes the bf16 kernel noise that otherwise dominates the comparison:
    # in bf16 the two torch versions pick different sdpa backends and disagree by
    # several logits on the largest entries even for identical arithmetic.
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--compare", nargs=2, metavar=("A.npz", "B.npz"))
    # bf16 logits of an 8B model land in the +-30 range; 0.35 is well inside the
    # accumulation noise of two different kernels and far below what a wrong
    # rotary schedule or a missing norm would produce (those show up in top-1).
    ap.add_argument("--atol", type=float, default=0.35)
    args = ap.parse_args()

    if args.compare:
        raise SystemExit(compare(args.compare[0], args.compare[1], args.atol))
    if not (args.impl and args.model and args.out):
        raise SystemExit("--impl, --model and --out are required unless --compare is given")
    run(args.impl, args.model, args.ref_model, args.n_tokens, args.out, args.device, args.dtype, args.attn)


if __name__ == "__main__":
    main()
