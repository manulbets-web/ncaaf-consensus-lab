#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/manulbets-web/ncaaf-consensus-lab.git"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

if ! command -v git >/dev/null 2>&1; then
  echo "git is not installed or not on PATH." >&2
  exit 1
fi

if [[ ! -d .git ]]; then
  git init
fi

git branch -M main
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REPO_URL"
else
  git remote add origin "$REPO_URL"
fi

git add .
if ! git diff --cached --quiet; then
  git commit -m "Deploy NCAAF Consensus Lab v3.5.9"
else
  echo "No new changes to commit."
fi

echo "Pushing to $REPO_URL"
git push -u origin main

echo
echo "GitHub push complete. In Posit Connect Cloud, publish from GitHub:"
echo "  repository: manulbets-web/ncaaf-consensus-lab"
echo "  branch:     main"
echo "  primary:    app.py"
