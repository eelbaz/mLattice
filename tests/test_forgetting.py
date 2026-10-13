#!/usr/bin/env python3
"""ACCEPTANCE TEST (TDD): published forgetting number.

Published: 0.998 -- after a key's value is OVERWRITTEN, queries must return the
UPDATED value (the delta rule's last-write-wins), not the original.

Primary protocol (the 131K forgetting benchmark):
  * N = 5,461 pairs stored with 14 filler tokens per pair,
  * then 2,730 of those keys re-presented with NEW values (~50%),
  * total (5,461 + 2,730) * 16 = 131,064 tokens + queries ~= 131K-class;
  * Q = 1,000 queries: 500 on UPDATED keys (must return the NEW value) and
    500 on UNTOUCHED keys (must still return the ORIGINAL value);
  * accuracy >= 0.998 on updated AND on untouched queries (1,000 queries ->
    two errors allowed overall).

Secondary: dense N=64 and N=1,024 (filler=14) overwrite checks.

Run:  python3 tests/test_forgetting.py [cuda|mps|cpu]
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from mulattice.protocol import SynthCfg, build_model, eval_batch, force_fast_scan, synth_batch

CKPT = os.environ.get("CKPT", os.path.join(HERE, "..", "checkpoints", "forgetting.pt"))
# CKPT env override: validate the test path against periodic checkpoints
TARGET = 0.998


def device_arg() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def run_case(name, n_pairs, filler, n_upd_q, n_keep_q, seed, dev) -> dict:
    """Store n_pairs, overwrite half, query n_upd_q updated + n_keep_q untouched keys."""
    if not os.path.exists(CKPT):
        return {"name": name, "error": f"checkpoint missing: {CKPT}"}
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = SynthCfg()
    if ck.get("proto_cfg", {}).get("addr_scale") is not None:
        cfg.addr_scale = float(ck["proto_cfg"]["addr_scale"])  # build at TRAINED scale
    m = build_model(cfg)
    m.load_state_dict(ck["model"])
    m = m.to(dev).eval()
    force_fast_scan(m)
    g = torch.Generator().manual_seed(seed)
    n_upd = n_pairs // 2
    # force a HALF updated / HALF untouched query split (clamped to pool sizes)
    perm = torch.randperm(n_pairs, generator=g)
    upd_idx = perm[:n_upd]
    upd_set = set(upd_idx.tolist())
    updated_pool = [i for i in perm.tolist() if i in upd_set]
    untouched_pool = [i for i in perm.tolist() if i not in upd_set]
    qpos = torch.tensor(updated_pool[:min(n_upd_q, len(updated_pool))]
                        + untouched_pool[:min(n_keep_q, len(untouched_pool))]).unsqueeze(0)
    b = synth_batch(1, n_pairs, qpos.shape[1], cfg, filler=filler,
                    overwrite_frac=0.5, generator=g, qpos=qpos,
                    updated_idx=upd_idx.unsqueeze(0))
    t0 = time.time()
    res = eval_batch(m, b, dev)
    res["dt"] = time.time() - t0
    res["T"] = b["L"]
    res["name"] = name
    return res


def main() -> int:
    dev = device_arg()
    print("=" * 74)
    print(f"FORGETTING acceptance test: store N, overwrite 50%, query updated+untouched")
    print(f"target >= {TARGET} on BOTH updated (new value) and untouched (original)")
    print("=" * 74)
    cases = [
        ("primary N=5,461 filler=14 (~131K tokens)", 5461, 14, 500, 500, 777001),
        ("secondary N=1,024 filler=14", 1024, 14, 300, 300, 777002),
        ("secondary N=64 dense", 64, 0, 256, 256, 777003),
    ]
    ok_all = True
    results = []
    for name, N, f, n_u, n_k, seed in cases:
        r = run_case(name, N, f, n_u, n_k, seed, dev)
        if "error" in r:
            print(f"[FAIL] {name}: {r['error']}")
            results.append(r)
            ok_all = False
            continue
        ok = r["acc_updated"] >= TARGET and r["acc_untouched"] >= TARGET
        ok_all &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: "
              f"updated {r['acc_updated']:.4f}  untouched {r['acc_untouched']:.4f}  "
              f"overall {r['acc']:.4f} ({r['correct']}/{r['n']})  T={r['T']}  [{r['dt']:.1f}s]")
        results.append(r)
    out = os.path.join(HERE, "..", "results", "test_forgetting.json")
    with open(out, "w") as fh:
        json.dump({"target": TARGET, "results": results}, fh, indent=2)
    print(f"results -> {out}")
    print(f"{'ALL PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
