#!/usr/bin/env python3
"""Gentle final polish of the capacity model at the exact acceptance distribution.

Contingency tool (fired manually when the curriculum's self-check lands below
0.999): the level fillers were [0,0,0,4,8] but the test uses filler=14; the
diagnostic sweep showed f=8 at 0.992 vs f=14 at 0.977 -- the gap is distribution
mismatch, not mechanism.  This script fine-tunes at f in {8,14} with a LOW entry
learning rate (no restart shock), probes on the exact test protocol every 250
steps, keeps the best checkpoint, and exits early on two consecutive >= 0.999
probes.
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
CKPT_IN = os.environ.get("RESUME", os.path.join(ROOT, "checkpoints", "capacity8k_p8192_best.pt"))
CKPT_OUT = os.environ.get("OUT", os.path.join(ROOT, "checkpoints", "capacity8k.pt"))
MAX_STEPS = int(os.environ.get("STEPS", "1600"))
LR = float(os.environ.get("LR", "2e-4"))
HEARTBEAT = os.path.join(ROOT, "logs", "polish_capacity.heartbeat")

CFG = SynthCfg()


def save(m, path, extra=None):
    live_scale = float(m.blocks[0].mixer.cfg.addr_scale)
    payload = {"model": m.state_dict(), "proto_cfg": {
        "NK": CFG.NK, "NV": CFG.NV, "d_model": CFG.d_model, "n_heads": CFG.n_heads,
        "d_k": CFG.d_k, "d_v": CFG.d_v, "addr_scale": live_scale,
        "retention_init": CFG.retention_init}, "cfg": "capacity polish at acceptance distribution"}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def main():
    dev = DEVICE
    ck = torch.load(CKPT_IN, map_location="cpu", weights_only=False)
    cfg = SynthCfg()
    if ck.get("proto_cfg", {}).get("addr_scale"):
        cfg.addr_scale = float(ck["proto_cfg"]["addr_scale"])
    m = build_model(cfg)
    m.load_state_dict(ck["model"])
    m = m.to(dev).train()
    force_fast_scan(m)
    print(f"polish from {CKPT_IN} (step {ck.get('step')}, acc {ck.get('acc')}, "
          f"scale {cfg.addr_scale:.0f}), lr {LR}, max {MAX_STEPS} steps", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_STEPS, eta_min=LR / 4)
    g = torch.Generator().manual_seed(11)
    best = 0.0
    good = 0
    t0 = time.time()
    hist = []
    fcyc = [8, 14, 14, 8, 14]        # eval-distribution heavy
    for s in range(MAX_STEPS):
        f = fcyc[s % len(fcyc)]
        Q = 128 if s % 3 == 0 else 64
        ow = 0.25 if s % 4 == 1 else 0.0
        b = synth_batch(1, 8192, Q, CFG, filler=f, overwrite_frac=ow, generator=g)
        tok = b["tokens"].to(dev)
        logits = logits_at(m, tok, b["ans_pos"].to(dev))
        gold = b["gold"].to(dev).reshape(-1)
        # DFT-style reward rectification (arXiv:2508.05629): per-position CE
        # weighted by 1/p(gold), clipped, so the few answers the model still
        # finds improbable dominate the gradient -- a one-line objective fix
        # aimed exactly at the last 0.2 points (uniform polish wobbles at the
        # optimum instead: measured 1.0000 -> 0.9980 -> 0.9961).
        p_gold = torch.softmax(logits.reshape(-1, CFG.V), -1).gather(1, gold.unsqueeze(1)).squeeze(1)
        w = (1.0 / p_gold.clamp_min(1e-4)).clamp(1.0, 10.0)
        ce_vec = F.cross_entropy(logits.reshape(-1, CFG.V), gold, reduction="none")
        ce = (w * ce_vec).sum() / w.sum()
        kp = b["key_pos"]
        if b["upd_key_pos"] is not None:
            kp = torch.cat([kp, b["upd_key_pos"]], 1)
        sl = spread_loss(m, tok, kp.to(dev), CFG, gate_subsample=1024)
        loss = ce + sl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if s % 50 == 0:
            print(f"  polish {s:4d} f{f} Q{Q} ce {float(ce.detach()):.4f} T{b['L']} "
                  f"{time.time()-t0:6.0f}s", flush=True)
            with open(HEARTBEAT, "w") as fh:
                fh.write(f"{s}\n")
        if (s + 1) % 250 == 0 or s == MAX_STEPS - 1:
            gg = torch.Generator().manual_seed(5)
            bb = synth_batch(1, 8192, 512, CFG, filler=14, generator=gg)
            r = eval_batch(m, bb, dev)
            dense = eval_batch(m, synth_batch(1, 8192, 512, CFG, filler=0,
                                              generator=torch.Generator().manual_seed(6)), dev)
            hist.append({"s": s, "probe": round(r["acc"], 4), "dense": round(dense["acc"], 4)})
            with open(os.path.join(ROOT, "results", "polish_capacity.json"), "w") as fh:
                json.dump(hist, fh, indent=2)
            print(f"  PROBE f14 {r['acc']:.4f} ({r['correct']}/{r['n']})  "
                  f"dense {dense['acc']:.4f}  [{time.time()-t0:.0f}s]", flush=True)
            if r["acc"] >= best:
                best = r["acc"]
                save(m, CKPT_OUT, {"step": s, "acc": r["acc"]})
            good = good + 1 if r["acc"] >= 0.999 else 0
            if good >= 2:
                print(f"TARGET: two consecutive probes >= 0.999 at step {s}", flush=True)
                save(m, CKPT_OUT, {"step": s, "acc": r["acc"]})
                return
    print(f"done; best probe {best:.4f}; best model saved to {CKPT_OUT}", flush=True)


if __name__ == "__main__":
    main()
