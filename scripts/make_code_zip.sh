#!/usr/bin/env bash
# Package the committed code (no data, no venv) as dist/p21_code.zip for manual upload
# to Colab/Kaggle when cloning from GitHub is not possible. Records the commit in p21/COMMIT.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "Uncommitted changes present; commit first so the zip matches a commit." >&2
  exit 1
fi
mkdir -p dist
commit=$(git rev-parse HEAD)
git archive --format=zip --prefix=p21/ --add-virtual-file="p21/COMMIT:${commit}" -o dist/p21_code.zip HEAD
echo "dist/p21_code.zip  ($(du -h dist/p21_code.zip | cut -f1), commit ${commit:0:7})"
