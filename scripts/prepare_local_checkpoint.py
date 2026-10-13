#!/usr/bin/env python
"""Materialize a Hub checkpoint as a local directory this repo's pinned stack can read.

Two things stand between a Hub id and `from_pretrained` here, neither about the
model weights:

1. Cache layout. On clusters whose compute nodes have no outbound network, a Hub
   id that resolves on the login node can fail on a worker when the transformers
   and huggingface_hub caches disagree. A plain local directory (weights symlinked,
   so no second copy) avoids both, and its basename becomes a clean MODEL_TAG.

2. tokenizer.json format. Files written by tokenizers >= 0.20 store BPE merges as
   ["a", "b"] pairs; tokenizers 0.19 (pinned by transformers 4.44) only parses the
   older "a b" string form and dies with
       Exception: data did not match any variant of untagged enum ModelWrapper
   The rewrite is applied only when needed and is checked for losslessness first
   (see models_ext/tokenizer_compat.py). A checkpoint already in the old format is
   copied through untouched.

Run once, from the repo root:
    python scripts/prepare_local_checkpoint.py --src Qwen/Qwen2.5-14B-Instruct \
        --out "$SCRATCH/models/Qwen2.5-14B-Instruct"
"""

import argparse
import json
import os
import shutil
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from models_ext.tokenizer_compat import downgrade_merges

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Hub id or local snapshot directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src = args.src
    if not os.path.isdir(src):
        from huggingface_hub import snapshot_download
        src = snapshot_download(src, ignore_patterns=["*.gguf", "*.pth"])

    out = args.out
    if os.path.isdir(out) and os.listdir(out) and not args.force:
        raise SystemExit(f"{out} exists and is not empty; pass --force to overwrite")
    os.makedirs(out, exist_ok=True)

    for name in sorted(os.listdir(src)):
        src_path = os.path.join(src, name)
        dst_path = os.path.join(out, name)
        if os.path.islink(dst_path) or os.path.exists(dst_path):
            os.remove(dst_path)
        if name.endswith(WEIGHT_SUFFIXES):
            # Symlink through to the blob: the shards are tens of GB and never edited.
            os.symlink(os.path.realpath(src_path), dst_path)
        elif name == "tokenizer.json":
            tok = json.load(open(src_path))
            n = downgrade_merges(tok)
            if n:
                # Keep the untouched original next to it, so the rewrite is auditable.
                shutil.copyfile(src_path, os.path.join(out, "tokenizer.json.upstream"))
                json.dump(tok, open(dst_path, "w"), ensure_ascii=False)
                print(f"tokenizer.json: rewrote {n} merges to the tokenizers<0.20 string form")
            else:
                shutil.copyfile(src_path, dst_path)
                print("tokenizer.json: already in the tokenizers<0.20 merge format, copied as-is")
        else:
            shutil.copyfile(src_path, dst_path)

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
