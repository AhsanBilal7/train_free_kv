#!/bin/bash
# =============================================================================
# Build RULER configs longer than the Hub's (simonjegou/ruler stops at 16384).
#
# The Hub export was generated with NVIDIA RULER as it stood on 2024-09-30
# (commit 3b8928f), the Llama-3.1-8B-Instruct tokenizer, --model_template_type
# base, seed 42 and wonderwords 2.2.0. This script reproduces that pipeline, so
# 65536 / 131072 are built by the same generator, seed and prompt wording as the
# 4k-16k numbers already in pred/. validate.py proves it: regenerating the 4096
# config this way matches the Hub rows byte-for-byte on all 13 tasks.
#
# One-time setup, on a LOGIN node (compute nodes have no network):
#   bash make_data.sh setup
# Generation (CPU only; submit it, do not run it on a login node):
#   sbatch -p cpu -t 12:00:00 -c 32 --mem=64G --account=$ACCOUNT \
#          --wrap "bash $PWD/make_data.sh generate 65536 131072"
# Check the pipeline against the Hub (cheap, 5 rows/task at 4096):
#   bash make_data.sh validate
#
# Output: $RULER_LOCAL_DIR/<length>/test-00000-of-00001.parquet + README.md,
# which pred.py loads with load_dataset($RULER_LOCAL_DIR, "<length>").
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
GEN_ROOT="${RULER_GEN_ROOT:-$REPO_ROOT/data/ruler_gen}"
OUT_DIR="${RULER_LOCAL_DIR:-$REPO_ROOT/data/ruler_long}"
RULER_REV="3b8928f"                     # NVIDIA/RULER HEAD on 2024-09-30
NUM_SAMPLES="${NUM_SAMPLES:-500}"       # per task, as in the Hub configs
TASKS=(niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2
       niah_multikey_3 niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2)

REPO="$GEN_ROOT/RULER_$RULER_REV"
VENV="$GEN_ROOT/venv"
SHIM="$GEN_ROOT/shim"

_env() {
    export NLTK_DATA="$GEN_ROOT/nltk_data"
    export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
    # prepare.py shells out to a bare `python`, so the venv must lead PATH.
    export PATH="$VENV/bin:$PATH"
    export PYTHONPATH="$SHIM${PYTHONPATH:+:$PYTHONPATH}"
    TOK="${RULER_TOKENIZER:-$(ls -d "$HF_HOME"/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*/ | head -1)}"
}

setup() {
    mkdir -p "$GEN_ROOT" && chmod 700 "$GEN_ROOT"   # nltk refuses group-writable dirs
    cd "$GEN_ROOT"
    [[ -d RULER_src ]] || git clone -q https://github.com/NVIDIA/RULER.git RULER_src
    [[ -d "$REPO" ]] || git -C RULER_src worktree add -q "$REPO" "$RULER_REV"
    if [[ ! -x "$VENV/bin/python" ]]; then
        uv venv -q --python 3.11 "$VENV"
        # wonderwords MUST be 2.2.0: 3.x changed the word lists, which changes every
        # niah key and every cwe word while leaving the other tasks identical.
        VIRTUAL_ENV="$VENV" uv pip install -q "transformers==4.44.2" tokenizers \
            sentencepiece nltk "wonderwords==2.2.0" pyyaml tqdm numpy scipy \
            html2text beautifulsoup4 tenacity pandas pyarrow jinja2
    fi
    mkdir -p -m 700 nltk_data
    "$VENV/bin/python" -c "import nltk; [nltk.download(x, download_dir='$GEN_ROOT/nltk_data', quiet=True) for x in ('punkt', 'punkt_tab')]"

    # The 2024 revision imports NeMo only for two jsonl helpers; shim them.
    local mu="$SHIM/nemo/collections/asr/parts/utils"
    mkdir -p "$mu"
    for d in nemo nemo/collections nemo/collections/asr nemo/collections/asr/parts "${mu#$SHIM/}"; do
        touch "$SHIM/$d/__init__.py"
    done
    cat > "$mu/manifest_utils.py" <<'EOF'
import json


def read_manifest(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_manifest(path, data, ensure_ascii=True):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=ensure_ascii) + "\n")
EOF

    # Source corpora. english_words.json is a Git LFS object in the repo, so a plain
    # clone holds only the pointer; fetch the blob itself.
    local js="$REPO/scripts/data/synthetic/json"
    cd "$js"
    [[ $(stat -c %s english_words.json) -gt 1000000 ]] || \
        curl -sfL -o english_words.json \
            "https://media.githubusercontent.com/media/NVIDIA/RULER/$RULER_REV/scripts/data/synthetic/json/english_words.json"
    [[ -s squad.json && -s hotpotqa.json ]] || bash download_qa_dataset.sh
    [[ -s PaulGrahamEssays.json ]] || "$VENV/bin/python" download_paulgraham_essay.py
    echo "setup done: $REPO"
}

gen_one() {   # <length> <task> -> raw jsonl under $GEN_ROOT/raw/<length>
    local L="$1" t="$2"
    cd "$REPO/scripts/data"
    python prepare.py --save_dir "$GEN_ROOT/raw/$L" --task "$t" \
        --tokenizer_path "$TOK" --tokenizer_type hf --max_seq_length "$L" \
        --model_template_type base --num_samples "$NUM_SAMPLES" \
        > "$GEN_ROOT/raw/log_${L}_${t}.txt" 2>&1
    local n; n=$(wc -l < "$GEN_ROOT/raw/$L/$t/validation.jsonl" 2>/dev/null || echo 0)
    echo "$L $t rows=$n"
    [[ "$n" -eq "$NUM_SAMPLES" ]]
}

generate() {
    _env
    mkdir -p "$GEN_ROOT/raw"
    local pids=() L t fail=0
    # One process per (length, task): prepare.py is single-threaded.
    for L in "$@"; do for t in "${TASKS[@]}"; do
        gen_one "$L" "$t" & pids+=($!)
    done; done
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    [[ $fail -eq 0 ]] || { echo "some tasks failed; see $GEN_ROOT/raw/log_*" >&2; exit 1; }
    for L in "$@"; do
        python "$HERE/convert_ruler.py" --raw_dir "$GEN_ROOT/raw/$L" --length "$L" --out_dir "$OUT_DIR"
    done
    python "$HERE/check_lengths.py" --data_dir "$OUT_DIR" --lengths "$@" --tokenizer "$TOK"
}

validate() {
    _env
    local d="$GEN_ROOT/validate"
    rm -rf "$d"
    cd "$REPO/scripts/data"
    for t in "${TASKS[@]}"; do
        python prepare.py --save_dir "$d/raw" --task "$t" --tokenizer_path "$TOK" \
            --tokenizer_type hf --max_seq_length 4096 --model_template_type base \
            --num_samples 5 > /dev/null 2>&1
    done
    python "$HERE/convert_ruler.py" --raw_dir "$d/raw" --length 4096 --out_dir "$d/hf"
    python "$HERE/validate.py" --local "$d/hf" --length 4096 \
        --hub "$(ls -d "$HF_HOME"/hub/datasets--simonjegou--ruler/snapshots/*/ | head -1)"
}

cmd="${1:-}"; shift || true
case "$cmd" in
    setup)    setup ;;
    generate) [[ $# -gt 0 ]] || set -- 65536 131072; generate "$@" ;;
    validate) validate ;;
    *) echo "usage: $0 setup | generate <len>... | validate" >&2; exit 2 ;;
esac
