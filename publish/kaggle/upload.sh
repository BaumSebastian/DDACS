#!/usr/bin/env bash
# Publish the DDACS teaser to Kaggle Datasets.
#
# Prereqs (once):
#   pip install kaggle
#   Auth is ambient via ~/.kaggle/access_token (already present on this machine).
#
# Set your username in publish/kaggle/dataset-metadata.json ("id" field) first.
#
# Usage:
#   ./publish/kaggle/upload.sh create      # first publish
#   ./publish/kaggle/upload.sh version     # update the existing dataset
set -euo pipefail

MODE="${1:-create}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # publish/kaggle
PUB="$(cd "$HERE/.." && pwd)"                           # publish/
STAGE="$PUB/.staging"

command -v kaggle >/dev/null 2>&1 || { echo "Kaggle CLI not found. Install: pip install kaggle" >&2; exit 1; }
if grep -q "INSERT_KAGGLE_USERNAME" "$HERE/dataset-metadata.json"; then
  echo "Edit publish/kaggle/dataset-metadata.json: set \"id\" to <your-username>/ddacs-teaser" >&2
  exit 1
fi

"$PUB/stage_teaser.sh"

ID="$(python3 -c "import json;print(json.load(open('$HERE/dataset-metadata.json'))['id'])")"

# --dir-mode zip packs the data/ subfolder into the upload.
case "$MODE" in
  create)
    # Kaggle requires dataset-metadata.json at the upload root.
    cp "$HERE/dataset-metadata.json" "$STAGE/dataset-metadata.json"
    kaggle datasets create -p "$STAGE" --dir-mode zip
    ;;
  version)
    # Kaggle rejects a version built on a stale snapshot ("cannot update of non
    # current metadata") — common after editing/making-public on the website.
    # Pull the server's current metadata, then overlay only our editable fields.
    kaggle datasets metadata "$ID" -p "$STAGE"
    python3 - "$STAGE/dataset-metadata.json" "$HERE/dataset-metadata.json" <<'PY'
import json, sys
cur = json.load(open(sys.argv[1]))
mine = json.load(open(sys.argv[2]))
for k in ("id", "title", "subtitle", "description", "keywords", "licenses", "resources"):
    if k in mine:
        cur[k] = mine[k]
json.dump(cur, open(sys.argv[1], "w"), indent=2, ensure_ascii=False)
print("merged editable fields into the current metadata")
PY
    kaggle datasets version -p "$STAGE" --dir-mode zip -m "Refine metadata (title, description, column docs)"
    ;;
  *) echo "usage: $0 [create|version]" >&2; exit 1 ;;
esac
