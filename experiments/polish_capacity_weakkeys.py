#!/usr/bin/env python3
"""Weak-key targeted polish (final 0.2 points).

Census (results/weak_keys.json, 2 seeds x 4,096 queries): ALL residual errors
at the 0.998 plateau come from ~16 distinct key ids out of 16,384 -- 0.1% of
the vocabulary.  The full-batch polish cannot move them (0.4% exposure per
step) and uniform training wobbles the optimum instead.

This script:
  * alternates small 512-pair batches whose stored keys and query set FORCE the
    weak keys (every batch stores AND queries all of them; DFT 1/p weighting
    on top) with full 8,192-pair stability batches;
  * probes on FRESH census seeds (never trained on) every 100 steps, f=14,
    Q=2,048, and saves when two consecutive probes show <= 1 error total.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from mulattice.protocol import (SynthCfg, build_model, eval_batch, force_fast_scan, logits_at,
                      spread_loss, synth_batch)

DEVICE = os.environ.get("DEVICE", "cuda")
CKPT_IN = os.environ.get("RESUME", os.path.join(ROOT, "checkpoints", "capacity8k.pt"))
CKPT_OUT = os.environ.get("OUT", os.path.join(ROOT, "checkpoints", "capacity8k.pt"))
MAX_STEPS = int(os.environ.get("STEPS", "500"))
LR = float(os.environ.get("LR", "5e-5"))
CFG = SynthCfg()


def save(m, path, extra=None):
    live_scale = float(m.blocks[0].mixer.cfg.addr_scale)
    payload = {"model": m.state_dict(), "proto_cfg": {
        "NK": CFG.NK, "NV": CFG.NV, "d_model": CFG.d_model, "n_heads": CFG.n_heads,
        "d_k": CFG.d_k, "d_v": CFG.d_v, "addr_scale": live_scale,
        "retention_init": CFG.retention_init}, "cfg": "weak-key targeted polish"}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def weighted_ce(logits, gold):
    p_gold = torch.softmax(logits, -1).gather(1, gold.unsqueeze(1)).squeeze(1)
    w = (1.0 / p_gold.clamp_min(1e-4)).clamp(1.0, 10.0)
    ce_vec = F.cross_entropy(logits, gold, reduction="none")
    return (w * ce_vec).sum() / w.sum()


def main():
    dev = DEVICE
    weak = json.load(open(os.path.join(ROOT, "results", "weak_keys.json")))
    weak_t = torch.tensor(sorted(weak))
    print(f"weak-key polish: {len(weak_t)} keys, lr {LR}, max {MAX_STEPS}", flush=True)
    ck = torch.load(CKPT_IN, map_location="cpu", weights_only=False)
    cfg = SynthCfg()
    if ck.get("proto_cfg", {}).get("addr_scale"):
        cfg.addr_scale = float(ck["proto_cfg"]["addr_scale"])
    m = build_model(cfg)
    m.load_state_dict(ck["model"])
    m = m.to(dev).train()
    force_fast_scan(m)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    g = torch.Generator().manual_seed(21)
    t0 = time.time()
    good = 0
    probe_seed = 301
    hist = []
    for s in range(MAX_STEPS):
        if s % 3 == 2:                       # stability: full-size batch
            b = synth_batch(1, 8192, 64, CFG, filler=14, generator=g)
        else:                                # targeted: small batch with weak keys
            # consistent by construction: include_keys guarantees the weak keys
            # are stored exactly once; qpos guarantees every weak key is queried
            nw = len(weak_t)
            qpos = torch.cat([torch.arange(nw),
                              nw + torch.randperm(512 - nw, generator=g)[: 128 - nw]
                              ]).unsqueeze(0)
            b = synth_batch(1, 512, 128, CFG, filler=14, generator=g, qpos=qpos,
                            kw={"include_keys": weak_t})
        tok = b["tokens"].to(dev)
        logits = logits_at(m, tok, b["ans_pos"].to(dev))
        gold = b["gold"].to(dev).reshape(-1)
        ce = weighted_ce(logits.reshape(-1, CFG.V), gold)
        sl = spread_loss(m, tok, b["key_pos"].to(dev), CFG, gate_subsample=512)
        loss = ce + sl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if s % 50 == 0:
            print(f"  wk {s:4d} ce {float(ce.detach()):.4f} T{b['L']} {time.time()-t0:6.0f}s",
                  flush=True)
        if (s + 1) % 100 == 0 or s == MAX_STEPS - 1:
            errs = 0
            for _ in range(2):               # fresh census probes, 2048 queries each
                gg = torch.Generator().manual_seed(probe_seed)
                probe_seed += 1
                bb = synth_batch(1, 8192, 1024, CFG, filler=14, generator=gg)
                m.eval()
                with torch.no_grad():
                    pred = logits_at(m, bb["tokens"].to(dev), bb["ans_pos"].to(dev)).argmax(-1).cpu()
                m.train()
                errs += int((pred != bb["gold"]).sum())
            hist.append({"s": s, "errs_2048": errs})
            with open(os.path.join(ROOT, "results", "polish_weakkeys.json"), "w") as fh:
                json.dump(hist, fh, indent=2)
            print(f"  CENSUS probe (2x1024 fresh queries): {errs} errors "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            good = good + 1 if errs <= 1 else 0
            if errs <= 3:                     # always keep the best-so-far
                save(m, CKPT_OUT, {"step": s, "census_errs": errs})
            if good >= 2:
                print(f"TARGET: two consecutive censuses <= 1 error at step {s}", flush=True)
                save(m, CKPT_OUT, {"step": s, "census_errs": errs})
                return
    print("done", flush=True)


if __name__ == "__main__":
    main()
