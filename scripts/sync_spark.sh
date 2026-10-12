#!/bin/bash
# Sync the working tree to a remote GPU host and verify checksums match.
# Usage: ./scripts/sync_spark.sh [host] [remote-path]
#   host defaults to spark-alpha, remote path defaults to ~/mLattice
set -uo pipefail
HOST="${1:-spark-alpha}"
REMOTE="${2:-\$HOME/mLattice}"
cd "$(dirname "$0")/.."
for d in src scripts tests experiments; do
  rsync -azc --delete --exclude '__pycache__' -e "ssh -o BatchMode=yes" "$d/" "$HOST:$REMOTE/$d/" \
    || { echo "rsync FAILED for $d"; exit 1; }
done
diffs=0
for d in src scripts tests experiments; do
  out=$(rsync -azc -n --itemize-changes --exclude '__pycache__' -e "ssh -o BatchMode=yes" \
        "$d/" "$HOST:$REMOTE/$d/" | grep -v '^\.d' | grep -v '__pycache__' || true)
  [ -n "$out" ] && { echo "DIFFERS in $d:"; echo "$out" | head -5; diffs=1; }
done
[ "$diffs" = "0" ] && echo "SYNC VERIFIED (no source differences)" || { echo "SYNC MISMATCH"; exit 1; }
