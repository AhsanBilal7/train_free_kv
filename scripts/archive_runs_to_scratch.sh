#!/bin/bash
# =============================================================================
# Move completed run folders out of the repo and onto scratch, leaving a symlink
# behind so eval.py / collect_results.py / the figure scripts keep working.
#
# Home is small and inode-limited on ACES; pred/ folders are bulky and immutable
# once a run has finished, so they belong on /scratch. The symlink means nothing
# downstream has to know they moved.
#
# Usage (from the repo root):
#   bash scripts/archive_runs_to_scratch.sh --pattern 'lorekv-ent-*'          # dry run
#   bash scripts/archive_runs_to_scratch.sh --pattern 'lorekv-ent-*' --apply  # do it
#
# Options:
#   --pattern GLOB   which run folders to move, matched against the folder NAME
#                    (default: 'lorekv-ent-*', the entropy-weighted family)
#   --exp NAME       restrict to one benchmark (LongBench | RULER | NIAH);
#                    repeatable. Default: all three.
#   --dest DIR       archive root (default: $SCRATCH/train_free_kv_archive)
#   --apply          actually move; without it the script only prints the plan
#
# Run folders are named lorekv-<cell>-<brule>-<sel>[-<model>]_B_<B>_chunk_<cs>_...
# (scripts/run_naming.py), so --pattern selects on the readable identity:
#   'lorekv-ent-*'            every entropy-weighted run, any budget rule/selection
#   'lorekv-*-unif-topk_*'    every core-configuration run, any reliability cell
#   'lorekv-dis-*-llama_*'    the Llama disagreement family
#   'lorekv-*_B_1024_*'       every B=1024 run
# Pre-rename folders (causal_lookahead_<tag>_...) are matched by their old names;
# run scripts/rename_pred_runs.py first so one pattern covers everything.
# =============================================================================
set -euo pipefail

PATTERN='lorekv-ent-*'
DEST="${SCRATCH:-/scratch/user/$USER}/train_free_kv_archive"
APPLY=0
EXPS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pattern) PATTERN="$2"; shift 2 ;;
        --dest)    DEST="$2";    shift 2 ;;
        --exp)     EXPS+=("$2"); shift 2 ;;
        --apply)   APPLY=1;      shift ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ ${#EXPS[@]} -eq 0 ]] && EXPS=(LongBench RULER NIAH)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

echo "=============================================================="
echo " archive_runs_to_scratch"
echo "   pattern    = $PATTERN"
echo "   benchmarks = ${EXPS[*]}"
echo "   dest       = $DEST"
echo "   mode       = $([[ $APPLY -eq 1 ]] && echo APPLY || echo 'DRY RUN (pass --apply to move)')"
echo "=============================================================="

total=0
for exp in "${EXPS[@]}"; do
    src_root="$REPO_ROOT/experiments/$exp/pred"
    [[ -d "$src_root" ]] || { echo "[$exp] no pred/ dir; skipping"; continue; }
    dst_root="$DEST/$exp/pred"

    # -maxdepth/-mindepth 1 -type d: only top-level run folders, never their contents.
    # -type d also excludes symlinks, so re-running is idempotent -- an already
    # archived folder is a symlink and will not match.
    mapfile -t runs < <(find "$src_root" -mindepth 1 -maxdepth 1 -type d -name "$PATTERN" | sort)
    echo "[$exp] ${#runs[@]} folder(s) match"
    [[ ${#runs[@]} -eq 0 ]] && continue

    [[ $APPLY -eq 1 ]] && mkdir -p "$dst_root"
    for run in "${runs[@]}"; do
        name="$(basename "$run")"
        size="$(du -sh "$run" 2>/dev/null | cut -f1)"
        if [[ $APPLY -eq 1 ]]; then
            if [[ -e "$dst_root/$name" ]]; then
                echo "  SKIP  $name (already at $dst_root/$name)"
                continue
            fi
            mv "$run" "$dst_root/$name"
            ln -s "$dst_root/$name" "$run"
            echo "  MOVED $name  ($size)  -> $dst_root/$name  [symlinked back]"
        else
            echo "  would move $name  ($size)"
        fi
        total=$((total + 1))
    done
done

echo "--------------------------------------------------------------"
if [[ $APPLY -eq 1 ]]; then
    echo "moved $total folder(s). Each is symlinked back into experiments/<exp>/pred/,"
    echo "so eval.py and the figure scripts need no change."
    echo "To undo one:  rm <link> && mv $DEST/<exp>/pred/<name> <link>"
else
    echo "$total folder(s) would move. Re-run with --apply to do it."
fi
