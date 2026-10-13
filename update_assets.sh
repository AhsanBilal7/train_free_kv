#!/bin/bash
# Copy the figures of the main branch into this project-page branch.
#
#   git checkout gh-pages && bash update_assets.sh && git commit -am "[site] Update figures" && git push
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p assets/paper
for f in lorekv_architecture.svg lorekv_architecture.png lorekv_mechanism.gif; do
    git show "main:assets/$f" > "assets/$f"
done
for f in $(git ls-tree --name-only main assets/paper/); do
    git show "main:$f" > "$f"
done
