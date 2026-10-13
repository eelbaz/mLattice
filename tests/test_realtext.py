#!/usr/bin/env python3
"""ACCEPTANCE TEST (TDD): real-data long-context retrieval at the 0.999 level.

Protocol (the 259K real-data benchmark, same mining as eval_realdata.py):
  * all 3,548 mined wikitext-103 key->bigram-successor pairs stored with 70
    filler tokens of REAL corpus text per pair -> T = 3,548*72 + 2*Q ~= 259K;
  * Q = 2,048 queries, stratified 512 each over store depth quartiles;
  * accuracy >= 0.999 OVERALL and >= 0.999 in EVERY depth quartile
    (512 queries per quartile: one error = 0.99805 -> that quartile fails).

Run:  python3 tests/test_realtext.py [cuda|mps|cpu]
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from mulattice.protocol import RealData, logits_at, stratified_qpos

CKPT = os.environ.get("CKPT", os.path.join(HERE, "..", "checkpoints", "realtext.pt"))
# CKPT env override: validate the test path against periodic checkpoints
CORPUS = os.path.join(HERE, "..", "data", "wikitext103_slice.txt")
TARGET = 0.999
Q = 2048
FILLER = 70          # 3548*(70+2) + 2*2048 = 259,552 tokens ~= 259K
P_CAP = 3548         # same pair cap as experiments/eval_realdata.py (mining yields 3,804)
STRATA = 4
SEEDS = (4242, 4243)  # two independent eval sets; both must pass


def device_arg() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def load_model(dev):
    from mulattice.models import MQARModel, MixerCfg
    if not os.path.exists(CKPT):
        return None, f"checkpoint missing: {CKPT}"
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sc = ck.get("proto_cfg", {})
    d_model = sc.get("d_model", 128)
    m = MQARModel(sc.get("V", 8190), kind="mulattice", n_layers=1,
                  cfg=MixerCfg(d_model=d_model, n_heads=sc.get("n_heads", 2),
                               d_k=sc.get("d_k", d_model // 2), d_v=sc.get("d_v", d_model // 2),
                               addr_scale=sc.get("addr_scale", 181.0),
                               learn_addr_scale=False, bound_address=True,
                               retention_init=9.0))
    m.load_state_dict(ck["model"])
    m = m.to(dev).eval()
    for blk in m.blocks:
        blk.mixer.scan.crossover = 1e9
    return m, None


def main() -> int:
    if not os.path.exists(CORPUS):
        print(f"SKIP: corpus not found at {CORPUS}")
        print(f"  The real-text evaluation requires data/wikitext103_slice.txt (~38MB).")
        print(f"  This file is not tracked in git. Build it with scripts/prepare_wikitext_slice.py.")
        return 0
    dev = device_arg()
    print("=" * 74)
    print(f"REAL-DATA 259K acceptance test: 3,548 real pairs, filler=70 real-text tokens,")
    print(f"{Q} depth-stratified queries, target >= {TARGET} overall AND per depth quartile")
    print(f"[device={dev}]")
    print("=" * 74)
    m, err = load_model(dev)
    rd = RealData(CORPUS)
    print(f"corpus: {rd.n_pairs} kv-pairs, filler stream {len(rd.FILLER)} tokens", flush=True)
    if err:
        print(f"[FAIL] {err}")
        with open(os.path.join(HERE, "..", "results", "test_realdata999.json"), "w") as fh:
            json.dump({"target": TARGET, "error": err}, fh, indent=2)
        return 1
    ok_all = True
    results = []
    P = min(P_CAP, rd.n_pairs)
    for seed in SEEDS:
        qpos = stratified_qpos(P, Q, STRATA, seed=seed)
        b = rd.batch(P, FILLER, Q, seed=seed, qpos_idx=qpos)
        t0 = time.time()
        with torch.no_grad():
            pred = logits_at(m, b["tokens"].to(dev), b["ans_pos"].to(dev)).argmax(-1).cpu()
        dt = time.time() - t0
        ok = (pred == b["gold"]).squeeze(0)
        per = Q // STRATA
        strata_acc = [round(ok[i * per:(i + 1) * per].float().mean().item(), 4) for i in range(STRATA)]
        overall = ok.float().mean().item()
        ok_seed = overall >= TARGET and all(a >= TARGET for a in strata_acc)
        ok_all &= ok_seed
        print(f"[{'PASS' if ok_seed else 'FAIL'}] seed {seed}: overall {overall:.4f} "
              f"({int(ok.sum())}/{Q})  depth quartiles {strata_acc}  T={b['L']}  [{dt:.1f}s]",
              flush=True)
        results.append({"seed": seed, "overall": overall, "strata": strata_acc,
                        "T": b["L"], "correct": int(ok.sum())})
    out = os.path.join(HERE, "..", "results", "test_realdata999.json")
    with open(out, "w") as fh:
        json.dump({"target": TARGET, "Q": Q, "filler": FILLER, "results": results}, fh, indent=2)
    print(f"results -> {out}")
    print(f"{'ALL PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
