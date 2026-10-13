"""Dataset loaders for the benchmark drivers and the analysis figures.

LongBench (Bai et al., ACL 2024):
    The Hub dataset ``THUDM/LongBench`` uses a loading script and one ``data.zip``.
    ``datasets>=3`` does not run loading scripts. :func:`load_longbench_split` thus
    extracts the archive one time to ``$HF_HOME/longbench_jsonl`` and reads the
    jsonl files. The rows and fields are the same as in the scripted build.

RULER (Hsieh et al., 2024):
    The 4K, 8K and 16K configs come from the Hub dataset ``simonjegou/ruler``.
    ``experiments/RULER/long_context/make_data.sh`` makes the 64K and 128K configs
    with the same schema. :func:`load_ruler_config` reads them from parquet.
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
    """Return the path of the local jsonl file of a LongBench task.

    The function first looks in ``dataset_path``. It then looks in the extracted
    copy of the Hub ``data.zip`` and extracts the archive if necessary. The
    extraction is atomic, so parallel jobs do not read a partial copy.

    Args:
        task: The LongBench task name, for example ``"qasper"``.
        dataset_path: Optional directory that contains the jsonl files.

    Returns:
        The file path, or None if only the Hub has the data.
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
    """Load one LongBench task as a ``datasets.Dataset``.

    The function uses the local jsonl file if it exists. Otherwise it loads the task
    from the Hub.
    """
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
    """Load one RULER sequence-length config (13 tasks) as a ``datasets.Dataset``.

    The function loads 4096, 8192 and 16384 from the Hub dataset. It loads other
    lengths from ``local_dir``. ``long_context/validate.py`` checks that the local
    generator gives the same rows as the Hub.

    Raises:
        FileNotFoundError: If a local length is not available.
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
