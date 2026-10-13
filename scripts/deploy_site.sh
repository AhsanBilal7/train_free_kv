#!/bin/bash
# Publish the project page (website/index.html + assets/) to the gh-pages branch.
#
#   bash scripts/deploy_site.sh            # commit and push gh-pages
#   NO_PUSH=1 bash scripts/deploy_site.sh  # build the branch locally only
#
# Then enable GitHub Pages for the gh-pages branch (Settings -> Pages); the site is
# served at https://<user>.github.io/<repo>/.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WORK="$(mktemp -d)"
trap 'git worktree remove --force "$WORK" >/dev/null 2>&1 || true; rm -rf "$WORK"' EXIT

if git show-ref --quiet refs/heads/gh-pages; then
    git worktree add -q "$WORK" gh-pages
else
    git worktree add -q --detach "$WORK"
    git -C "$WORK" checkout -q --orphan gh-pages
fi
git -C "$WORK" rm -rq --ignore-unmatch . >/dev/null 2>&1 || true
find "$WORK" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +

cp website/index.html "$WORK/index.html"
mkdir -p "$WORK/assets/paper"
cp assets/lorekv_architecture.svg assets/lorekv_architecture.png assets/lorekv_mechanism.gif "$WORK/assets/"
cp assets/paper/*.png "$WORK/assets/paper/"
touch "$WORK/.nojekyll"

git -C "$WORK" add -A
if git -C "$WORK" diff --cached --quiet; then
    echo "gh-pages is up to date"
else
    git -C "$WORK" commit -q -m "[site] Publish the project page from $(git rev-parse --short HEAD)"
    echo "committed gh-pages"
fi
[[ -n "${NO_PUSH:-}" ]] || git -C "$WORK" push -q origin gh-pages
