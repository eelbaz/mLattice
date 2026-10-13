#!/usr/bin/env python3
"""Build data/wikitext103_slice.txt for the real-text benchmark.

The slice is the leading non-empty lines of the WikiText-103 raw training
split, up to the first line that brings the text to 40,000,000 characters
(40,076,535 bytes), taken from the first parquet shard of Salesforce/wikitext
on the Hugging Face Hub (pinned revision). It is byte-identical to the corpus
in the demo Space; RealData mines 3,804 key->bigram-successor pairs from it.

Run:  python3 scripts/prepare_wikitext_slice.py
Requires: huggingface_hub, pyarrow
"""
from __future__ import annotations

import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "wikitext103_slice.txt")
REPO = "Salesforce/wikitext"
REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
SHARD = "wikitext-103-raw-v1/train-00000-of-00002.parquet"
N_CHARS = 40_000_000
SHA256 = "9778e5ecca59f8f0f9b7866d67d8b078bb81f9d52ce95a44e868702456741fbf"


def main() -> int:
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("requires: pip install huggingface_hub pyarrow")
        return 1
    shard = hf_hub_download(REPO, SHARD, repo_type="dataset", revision=REVISION)
    lines, n = [], 0
    for line in pq.read_table(shard, columns=["text"])["text"].to_pylist():
        if not line.strip():
            continue
        lines.append(line)
        n += len(line)
        if n >= N_CHARS:
            break
    data = "".join(lines).encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        print(f"sha256 mismatch: got {digest}, expected {SHA256}")
        return 1
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as fh:
        fh.write(data)
    print(f"wrote {os.path.normpath(OUT)} ({len(data):,} bytes, sha256 {digest})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
