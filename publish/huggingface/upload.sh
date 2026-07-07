#!/usr/bin/env bash
# Publish the DDACS teaser to Hugging Face Datasets.
#
# Prereqs (once):
#   pip install huggingface_hub
#   hf auth login                 # (or: huggingface-cli login) — write token
#
# Usage:
#   HF_REPO=your-namespace/ddacs-teaser ./publish/huggingface/upload.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # publish/huggingface
PUB="$(cd "$HERE/.." && pwd)"                           # publish/
STAGE="$PUB/.staging"
REPO="${HF_REPO:-}"

[ -n "$REPO" ] || { echo "Set HF_REPO=your-namespace/ddacs-teaser" >&2; exit 1; }

# Prefer the new 'hf' CLI, fall back to the legacy 'huggingface-cli'.
if command -v hf >/dev/null 2>&1; then HFCLI=(hf)
elif command -v huggingface-cli >/dev/null 2>&1; then HFCLI=(huggingface-cli)
else echo "Hugging Face CLI not found. Install: pip install huggingface_hub" >&2; exit 1; fi

ROOT="$(cd "$PUB/.." && pwd)"                          # repo root

"$PUB/stage_teaser.sh"

# HF has no separate "kernels" concept and does not mangle zips, so bundle the
# tutorials directly. On a cloned HF repo the data/ + notebooks/ layout matches
# the source repo, so the notebooks run against '../data' unmodified.
mkdir -p "$STAGE/notebooks"
cp "$ROOT/notebooks/"*.ipynb   "$STAGE/notebooks/"
cp "$ROOT/notebooks/README.md" "$STAGE/notebooks/README.md"

# The HF dataset card = YAML front matter (card-header.md) + the shared teaser
# README body (single source), assembled here so the body is never duplicated.
cat "$HERE/card-header.md" "$PUB/teaser/README.md" > "$STAGE/README.md"

echo "Uploading $STAGE -> https://huggingface.co/datasets/$REPO"
"${HFCLI[@]}" upload "$REPO" "$STAGE" . --repo-type dataset \
  --commit-message "Publish DDACS teaser"
echo "Done: https://huggingface.co/datasets/$REPO"
