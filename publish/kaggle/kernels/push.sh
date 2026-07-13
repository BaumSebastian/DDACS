#!/usr/bin/env bash
# Publish the six DDACS tutorials as Kaggle Notebooks (Code tab), each attached
# to the baumsebastian/ddacs-teaser dataset.
#
# Prereqs: the ddacs-teaser dataset already exists on Kaggle, and `kaggle` is on
# PATH (auth via ~/.kaggle/access_token). Kernels are pushed PRIVATE — review
# each on the site, then make public.
#
# Usage:  ./publish/kaggle/kernels/push.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # publish/kaggle/kernels
REPO="$(cd "$HERE/../../.." && pwd)"                      # repo root
OUT="$REPO/publish/.staging/kernels"

command -v kaggle >/dev/null 2>&1 || { echo "Kaggle CLI not found. Install it (uv tool install ...)." >&2; exit 1; }

python3 "$HERE/build.py"

# Kaggle caps concurrent batch sessions at 5; pushing 6 kernels back to back can
# hit "Maximum batch CPU session count reached". The CLI exits 0 even then, so
# detect the error in the output and retry with a pause.
for dir in "$OUT"/*/; do
  echo "=== kaggle kernels push -p $dir ==="
  for attempt in 1 2 3 4 5; do
    out="$(kaggle kernels push -p "$dir" 2>&1)"; echo "$out"
    echo "$out" | grep -qi "error" || break
    # only the session cap is transient — anything else will not fix itself
    echo "$out" | grep -qi "Maximum batch" || { echo "non-retryable push error on $dir" >&2; exit 1; }
    [ "$attempt" -lt 5 ] || { echo "giving up on $dir" >&2; exit 1; }
    echo "session cap hit (attempt $attempt) — waiting 180s for a free slot"
    sleep 180
  done
done

echo "All kernels pushed (private). Review at https://www.kaggle.com/${USER:-baumsebastian}/code and set them public."
