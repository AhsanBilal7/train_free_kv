"""Dataset loaders shared by the benchmark drivers and the analysis figures.

LongBench (Bai et al., ACL 2024; https://huggingface.co/datasets/THUDM/LongBench)
    The Hub repository is a loading-script dataset (``LongBench.py`` plus one
    ``data.zip`` with every sub-task as jsonl). ``datasets>=3`` no longer runs
    loading scripts, so :func:`load_longbench_split` extracts the archive once
    into ``$HF_HOME/longbench_jsonl`` and reads the jsonl directly. Rows and fields
    are identical to the scripted build.

RULER (Hsieh et al., 2024; https://github.com/NVIDIA/RULER)
    4K/8K/16K come from the Hub mirror ``simonjegou/ruler`` (the kvpress release of
    RULER). 64K/128K are generated locally with
    ``experiments/RULER/long_context/make_data.sh`` in the same schema and read
    from parquet by :func:`load_ruler_config`.
"""
from __future__ import annotations

import glob
import os
import shutil
import tempfile
import zipfile

from datasets import load_dataset

# ---------------------------------------------------------------------------
# LongBench
# ---------------------------------------------------------------------------
# The 16 English LongBench tasks evaluated in the paper (Table 5), grouped as in
# the results tables.
LONGBENCH_TASKS = [
    "narrativeqa", "qasper", "multifieldqa_en",          # single-document QA
    "hotpotqa", "2wikimqa", "musique",                   # multi-document QA
    "gov_report", "qmsum", "multi_news",                 # summarization
    "trec", "triviaqa", "samsum",                        # few-shot learning
    "passage_count", "passage_retrieval_en",             # synthetic
    "lcc", "repobench-p",                                # code
]

# Tasks whose prompts are not wrapped in a chat template (LongBench convention:
# chat models are better off without it on few-shot and code completion).
LONGBENCH_NO_CHAT = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}


def _hf_cache_roots():
    roots = [os.environ.get("HF_DATASETS_CACHE"), os.environ.get("HF_HOME"),
             os.path.join(os.path.expanduser("~"), ".cache", "huggingface")]
    return [r for r in roots if r]


def _find_longbench_zip():
    hub = os.environ.get("HF_HUB_CACHE")
    hubs = ([hub] if hub else []) + [os.path.join(r, "hub") for r in _hf_cache_roots()]
    for h in hubs:
        hits = sorted(glob.glob(os.path.join(
            h, "datasets--THUDM--LongBench", "snapshots", "*", "data.zip")))
        if hits:
            return hits[0]
    return None


def longbench_jsonl(task, dataset_path=None):
    """Local ``<task>.jsonl`` for a LongBench sub-task, or None if only the Hub has it.

    Looks in ``dataset_path`` (a directory of jsonl files, or its parent) first,
    then in the extracted copy of the Hub snapshot's ``data.zip``, extracting it on
    first use. Extraction is atomic, so concurrent jobs cannot see a partial copy.
    """
    if dataset_path and os.path.isdir(dataset_path):
        for cand in (os.path.join(dataset_path, f"{task}.jsonl"),
                     os.path.join(dataset_path, "data", f"{task}.jsonl")):
            if os.path.exists(cand):
                return cand
    out_dir = os.path.join(_hf_cache_roots()[0], "longbench_jsonl")
    cand = os.path.join(out_dir, f"{task}.jsonl")
    if os.path.exists(cand):
        return cand
    zip_path = _find_longbench_zip()
    if zip_path is None:
        return None
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=".longbench_", dir=os.path.dirname(out_dir))
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        try:
            os.replace(os.path.join(tmp, "data"), out_dir)
        except OSError:
            pass                                  # a concurrent job extracted it first
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return cand if os.path.exists(cand) else None


def load_longbench_split(dataset_path, task):
    """One LongBench sub-task as a ``datasets.Dataset``, preferring the local jsonl."""
    jsonl = longbench_jsonl(task, dataset_path)
    if jsonl is not None:
        return load_dataset("json", data_files=jsonl, split="train")
    # Hub / arrow-cache path; data_dir reproduces the hashed offline cache names.
    return load_dataset(dataset_path, task, split="test", data_dir=f"{dataset_path}/data")


# ---------------------------------------------------------------------------
# RULER
# ---------------------------------------------------------------------------
RULER_HUB_LENGTHS = {"4096", "8192", "16384"}


def load_ruler_config(hub_dataset, local_dir, length):
    """One RULER sequence-length config (all 13 tasks) as a ``datasets.Dataset``.

    Lengths shipped by the Hub mirror are loaded from it; longer ones must have been
    generated into ``local_dir`` by ``experiments/RULER/long_context/make_data.sh``
    (``long_context/validate.py`` checks that the local generator reproduces the Hub
    rows byte for byte).
    """
    length = str(length)
    if length in RULER_HUB_LENGTHS:
        return load_dataset(hub_dataset, length, split="test")
    path = os.path.join(local_dir or "", length, "test-00000-of-00001.parquet")
    if not local_dir or not os.path.exists(path):
        raise FileNotFoundError(
            f"RULER config {length} is not on the Hub ({sorted(RULER_HUB_LENGTHS, key=int)}) "
            f"and no local copy exists at {path!r}. Generate it with "
            "experiments/RULER/long_context/make_data.sh and set RULER_LOCAL_DIR.")
    return load_dataset("parquet", data_files={"test": path}, split="test")
