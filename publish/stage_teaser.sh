#!/usr/bin/env bash
# Assemble the teaser bundle under publish/.staging/.
#
# Layout mirrors the repo (data/ + notebooks/) so the notebooks' hard-coded
# DATA_DIR = '../data' keeps working with no edits:
#
#   .staging/
#     data/{metadata.json, process_parameters.csv, h5/258864.zip}
#     ddacs_documentation.pdf
#     README.md
#
# Notebooks are published separately as Kaggle kernels (publish/kaggle/kernels),
# so they are intentionally not bundled into the dataset.
#
# Source dir holding metadata.json + process_parameters.csv + h5/<id>.zip.
# Defaults to ./data; override with DDACS_TEASER_SRC when that symlink is stale.
# Files are dereferenced on copy. Nothing here is committed — publish/ is ignored.
set -euo pipefail

PUB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # publish/
ROOT="$(cd "$PUB/.." && pwd)"                          # repo root
STAGE="$PUB/.staging"
SAMPLE_ID=258864
SRC="${DDACS_TEASER_SRC:-$ROOT/data}"

rm -rf "$STAGE"
mkdir -p "$STAGE/data/h5"

# 1. Data bundle — require the three core files to resolve under $SRC.
missing=()
for rel in metadata.json process_parameters.csv "h5/${SAMPLE_ID}.zip"; do
  [ -e "$SRC/$rel" ] || missing+=("$rel")
done
if [ "${#missing[@]}" -ne 0 ]; then
  echo "Missing under SRC='$SRC': ${missing[*]}" >&2
  echo "Point DDACS_TEASER_SRC at a dir with metadata.json, process_parameters.csv, h5/${SAMPLE_ID}.zip" >&2
  echo "(e.g. DDACS_TEASER_SRC=<data source dir> $0)" >&2
  exit 1
fi
echo "Staging data from $SRC ..."
cp -L "$SRC/metadata.json"          "$STAGE/data/metadata.json"
cp -L "$SRC/process_parameters.csv" "$STAGE/data/process_parameters.csv"
cp -L "$SRC/h5/${SAMPLE_ID}.zip"    "$STAGE/data/h5/${SAMPLE_ID}.zip"

# Documentation PDF — include it when present in the source dir.
if [ -e "$SRC/ddacs_documentation.pdf" ]; then
  cp -L "$SRC/ddacs_documentation.pdf" "$STAGE/ddacs_documentation.pdf"
  echo "Included ddacs_documentation.pdf"
else
  echo "note: ddacs_documentation.pdf not in $SRC — skipping" >&2
fi

# 2. Bundle-level readme (HF's upload overwrites this with the dataset card).
cp "$PUB/teaser/README.md" "$STAGE/README.md"

echo "Staged teaser at $STAGE"
du -sh "$STAGE"
