#!/usr/bin/env python3
"""ACCEPTANCE TEST (TDD): published capacity number.

Published: 0.999 recall with 8,192 pairs inside a ~131K-token sequence.

Protocol (the 131K capacity benchmark):
  * 8,192 distinct key->value pairs stored with 14 filler tokens before each
    pair -> 8,192 * 16 + 2*Q tokens ~= 131K-class sequence;
  * filler tokens are drawn from the key vocabulary EXCLUDING the stored keys
    (a filler token can never be confused with a store event);
  * Q = 1,024 queries at the end (so ">= 0.999" is statistically resolvable:
    one error in 1,024 = 0.99902 -> PASS, two errors = FAIL);
  * the answer is read causally at each query's SEP token;
  * accuracy must be >= 0.999.

Secondary: the DENSE protocol (no filler, 8,192 pairs, T ~= 18K) must also be
>= 0.999.

Run:  python3 tests/test_capacity.py [cuda|mps|cpu]
Exits 0 on PASS, 1 on FAIL.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from mulattice.protocol import (SynthCfg, build_model, eval_batch, force_fast_scan, synth_batch)

CKPT = os.environ.get("CKPT", os.path.join(HERE, "..", "checkpoints", "capacity8k.pt"))
# CKPT env override: validate the test path against periodic checkpoints
TARGET = 0.999
N_PAIRS = 8192
FILLER = 14          # 8192*(14+2) + 2*1024 = 133,120 tokens (~131K protocol)
NQ = 1024
SEED = 20261102


def device_arg() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def run(protocol_name: str, filler: int, dev: str) -> dict:
    if not os.path.exists(CKPT):
        return {"name": protocol_name, "error": f"checkpoint missing: {CKPT}"}
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    saved_cfg = ck.get("proto_cfg", {})
    cfg = SynthCfg()
    if saved_cfg.get("addr_scale") is not None:
        cfg.addr_scale = float(saved_cfg["addr_scale"])   # build at the TRAINED scale
    torch.manual_seed(SEED)
    m = build_model(cfg)
    if saved_cfg.get("NK") not in (None, cfg.NK) or saved_cfg.get("d_model") not in (None, cfg.d_model):
        return {"name": protocol_name, "error": f"checkpoint cfg mismatch: {saved_cfg}"}
    m.load_state_dict(ck["model"])
    m = m.to(dev).eval()
    force_fast_scan(m)
    g = torch.Generator().manual_seed(SEED)
    b = synth_batch(1, N_PAIRS, NQ, cfg, filler=filler, generator=g)
    t0 = time.time()
    res = eval_batch(m, b, dev)
    res["dt"] = time.time() - t0
    res["T"] = b["L"]
    res["name"] = protocol_name
    # depth breakdown: how accuracy varies with store depth (early/mid/late pairs)
    ok = None
    with torch.no_grad():
        from mulattice.protocol import logits_at
        pred = logits_at(m, b["tokens"].to(dev), b["ans_pos"].to(dev)).argmax(-1).cpu()
    ok = (pred == b["gold"]).squeeze(0)
    quart = NQ // 4
    res["depth_quartiles"] = [round(ok[i * quart:(i + 1) * quart].float().mean().item(), 4)
                              for i in range(4)]
    return res


def main() -> int:
    dev = device_arg()
    print("=" * 74)
    print(f"CAPACITY acceptance test: {N_PAIRS} pairs, {FILLER} filler/pair "
          f"(~131K tokens), {NQ} queries, target >= {TARGET}   [device={dev}]")
    print("=" * 74)
    results = []
    ok_all = True
    for name, f in (("131K protocol (filler=14)", FILLER), ("dense protocol (filler=0)", 0)):
        r = run(name, f, dev)
        if "error" in r:
            print(f"[FAIL] {name}: {r['error']}")
            results.append(r)
            ok_all = False
            continue
        ok = r["acc"] >= TARGET
        ok_all &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: acc {r['acc']:.4f} "
              f"({r['correct']}/{r['n']})  T={r['T']}  [{r['dt']:.1f}s]  "
              f"depth quartiles {r['depth_quartiles']}")
        results.append(r)
    out = os.path.join(HERE, "..", "results", "test_capacity.json")
    with open(out, "w") as fh:
        json.dump({"target": TARGET, "n_pairs": N_PAIRS, "results": results}, fh, indent=2)
    print(f"results -> {out}")
    print(f"{'ALL PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
