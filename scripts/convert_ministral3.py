#!/usr/bin/env python
"""Extract the text model of a Mistral3 checkpoint into a separate checkpoint.

``mistralai/Ministral-3-8B-Instruct-2512[-BF16]`` is a
``Mistral3ForConditionalGeneration`` checkpoint: a Pixtral vision tower, a
multimodal projector and the ``ministral3`` text model. The text weights have the
prefix ``language_model.``, and the text config is ``config.text_config``.

The eviction pipeline uses text only and loads each model with
``AutoModelForCausalLM.from_pretrained(MODEL_PATH)``. This script writes a
``Ministral3ForCausalLM`` checkpoint one time, so MODEL_PATH is a normal local
directory.

The script copies the tensors without change. Only the key names change::

    language_model.model.*                   ->  model.*
    language_model.lm_head.weight            ->  lm_head.weight
    vision_tower.*, multi_modal_projector.*  ->  removed

Usage::

    python scripts/convert_ministral3.py \\
        --src mistralai/Ministral-3-8B-Instruct-2512-BF16 \\
        --out models/Ministral-3-8B-Instruct-2512
"""

import argparse
import json
import os
import shutil
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from models_ext.tokenizer_compat import downgrade_merges

from safetensors import safe_open
from safetensors.torch import save_file

TEXT_PREFIX = "language_model."
DROP_PREFIXES = ("vision_tower.", "multi_modal_projector.")
# Everything a tokenizer needs, plus the files that document where this came from.
SIDECAR_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "tekken.json", "chat_template.jinja", "SYSTEM_PROMPT.txt", "params.json",
)


def resolve_source(src):
    """Return a local directory unchanged, else the Hub snapshot directory (cached files are used)."""
    if os.path.isdir(src):
        return src
    from huggingface_hub import snapshot_download
    return snapshot_download(src, ignore_patterns=["consolidated.safetensors", "*.pth", "*.gguf"])


def build_text_config(wrapper_config):
    """Return the nested text_config as a separate Ministral3Config dict."""
    text = dict(wrapper_config["text_config"])
    text["model_type"] = "ministral3"
    text["architectures"] = ["Ministral3ForCausalLM"]
    text["torch_dtype"] = wrapper_config.get("dtype", wrapper_config.get("torch_dtype", "bfloat16"))
    text["tie_word_embeddings"] = wrapper_config.get("tie_word_embeddings", False)
    # Provenance: which repo and which wrapper this text tower was carved out of.
    text["_extracted_from"] = {
        "architectures": wrapper_config.get("architectures"),
        "model_type": wrapper_config.get("model_type"),
        "transformers_version": wrapper_config.get("transformers_version"),
    }
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="mistralai/Ministral-3-8B-Instruct-2512-BF16",
                    help="Hub id or local directory of the Mistral3 checkpoint")
    ap.add_argument("--out", required=True, help="output directory for the text-only checkpoint")
    ap.add_argument("--force", action="store_true", help="overwrite a non-empty output directory")
    args = ap.parse_args()

    src = resolve_source(args.src)
    out = args.out
    if os.path.isdir(out) and os.listdir(out) and not args.force:
        raise SystemExit(f"{out} exists and is not empty; pass --force to overwrite")
    os.makedirs(out, exist_ok=True)

    wrapper_config = json.load(open(os.path.join(src, "config.json")))
    if wrapper_config.get("model_type") != "mistral3":
        raise SystemExit(f"expected a mistral3 wrapper config, got model_type={wrapper_config.get('model_type')!r}")
    if "quantization_config" in wrapper_config:
        raise SystemExit(
            "this checkpoint is quantized (%s); convert the -BF16 repo instead"
            % wrapper_config["quantization_config"].get("quant_method")
        )

    index_path = os.path.join(src, "model.safetensors.index.json")
    if os.path.exists(index_path):
        shard_files = sorted(set(json.load(open(index_path))["weight_map"].values()))
    else:
        shard_files = ["model.safetensors"]

    new_weight_map, total_bytes, n_kept, n_dropped = {}, 0, 0, 0
    for shard_i, shard in enumerate(shard_files, start=1):
        out_shard = f"model-{shard_i:05d}-of-{len(shard_files):05d}.safetensors"
        tensors = {}
        with safe_open(os.path.join(src, shard), framework="pt") as f:
            for key in f.keys():
                if key.startswith(DROP_PREFIXES):
                    n_dropped += 1
                    continue
                if not key.startswith(TEXT_PREFIX):
                    raise SystemExit(f"unexpected key outside the text tower and the dropped towers: {key}")
                new_key = key[len(TEXT_PREFIX):]
                t = f.get_tensor(key)
                tensors[new_key] = t
                new_weight_map[new_key] = out_shard
                total_bytes += t.numel() * t.element_size()
                n_kept += 1
        if not tensors:
            continue
        # `format: pt` is what transformers checks before loading a safetensors shard.
        save_file(tensors, os.path.join(out, out_shard), metadata={"format": "pt"})
        print(f"wrote {out_shard}: {len(tensors)} tensors", flush=True)
        del tensors

    json.dump(
        {"metadata": {"total_size": total_bytes}, "weight_map": new_weight_map},
        open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=2,
    )
    json.dump(build_text_config(wrapper_config), open(os.path.join(out, "config.json"), "w"), indent=2)

    gen_path = os.path.join(src, "generation_config.json")
    if os.path.exists(gen_path):
        gen = json.load(open(gen_path))
        # `max_length` from the wrapper is the 262144 context window, not a generation
        # cap; leaving it in makes `generate()` silently ignore max_new_tokens limits
        # set elsewhere, so drop it and let each caller pass its own.
        gen.pop("max_length", None)
        json.dump(gen, open(os.path.join(out, "generation_config.json"), "w"), indent=2)

    for name in SIDECAR_FILES:
        p = os.path.join(src, name)
        if not os.path.exists(p):
            continue
        if name == "tokenizer.json":
            # Ministral-3's file also uses the tokenizers>=0.20 merge format, which
            # the pinned tokenizers 0.19 cannot parse. Keep the original alongside.
            tok = json.load(open(p))
            n = downgrade_merges(tok)
            shutil.copyfile(p, os.path.join(out, "tokenizer.json.upstream"))
            json.dump(tok, open(os.path.join(out, name), "w"), ensure_ascii=False)
            print(f"tokenizer.json: rewrote {n} merges to the tokenizers<0.20 string form")
        else:
            shutil.copyfile(p, os.path.join(out, name))

    print(f"\nkept {n_kept} text tensors, dropped {n_dropped} vision/projector tensors")
    print(f"total_size = {total_bytes / 1e9:.2f} GB")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
