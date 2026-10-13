#!/usr/bin/env python3
"""Weak-key targeted polish for the FORGETING task (endgame, mirrors
polish_weakkeys.py which closed capacity's last 0.2 points).

Census: probe the exact acceptance protocol (N=5,461, f=14, ow=0.5) on fresh
seeds; collect (a) keys whose UPDATED value is returned wrong, and (b) keys
whose ORIGINAL value is returned wrong after nearby overwrites.  Then train
small consistent batches (protocol.synth_batch include_keys + explicit qpos
hitting every weak key, updated_idx covering the weak-updated set) with
per-position 1/p CE weighting, alternating with full-size stability batches.
Saves when a fresh-seed census improves; exits on two consecutive censuses
with <= 2 total errors (2x1000 queries => >=0.999 average).
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

from mulattice.protocol import (SynthCfg, build_model, force_fast_scan, logits_at,
                      spread_loss, synth_batch)

DEVICE = os.environ.get("DEVICE", "cuda")
CKPT_IN = os.environ.get("RESUME", os.path.join(ROOT, "checkpoints", "forgetting.pt"))
CKPT_OUT = os.environ.get("OUT", os.path.join(ROOT, "checkpoints", "forgetting.pt"))
MAX_STEPS = int(os.environ.get("STEPS", "500"))
LR = float(os.environ.get("LR", "3e-5"))
CFG = SynthCfg()
N_PROTOCOL, F_PROTOCOL = 5461, 14


def save(m, path, extra=None):
    live_scale = float(m.blocks[0].mixer.cfg.addr_scale)
    payload = {"model": m.state_dict(), "proto_cfg": {
        "NK": CFG.NK, "NV": CFG.NV, "d_model": CFG.d_model, "n_heads": CFG.n_heads,
        "d_k": CFG.d_k, "d_v": CFG.d_v, "addr_scale": live_scale,
        "retention_init": CFG.retention_init}, "cfg": "forgetting weak-key polish"}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def weighted_ce(logits, gold):
    p_gold = torch.softmax(logits, -1).gather(1, gold.unsqueeze(1)).squeeze(1)
    w = (1.0 / p_gold.clamp_min(1e-4)).clamp(1.0, 10.0)
    ce_vec = F.cross_entropy(logits, gold, reduction="none")
    return (w * ce_vec).sum() / w.sum()


@torch.no_grad()
def census(m, seed):
    """Returns (n_err, weak_updated_keys, weak_untouched_keys) on fresh seeds."""
    g = torch.Generator().manual_seed(seed)
    n_upd = N_PROTOCOL // 2
    perm = torch.randperm(N_PROTOCOL, generator=g)
    upd_idx = perm[:n_upd]
    upd_set = set(upd_idx.tolist())
    pools_u = [i for i in perm.tolist() if i in upd_set]
    pools_k = [i for i in perm.tolist() if i not in upd_set]
    qpos = torch.tensor(pools_u[:500] + pools_k[:500]).unsqueeze(0)
    b = synth_batch(1, N_PROTOCOL, 1000, CFG, filler=F_PROTOCOL, overwrite_frac=0.5,
                    generator=g, qpos=qpos, updated_idx=upd_idx.unsqueeze(0))
    pred = logits_at(m, b["tokens"].to(DEVICE), b["ans_pos"].to(DEVICE)).argmax(-1).cpu()
    ok = (pred == b["gold"]).squeeze(0)
    err = ~ok
    # map failing query slots back to stored keys, SPLIT BY CLASS: the first 500
    # queries are on updated keys, the last 500 on untouched keys.  The prior
    # version merged the classes and trained every weak key as an UPDATED one --
    # but the failing class is untouched keys damaged by nearby overwrites, so
    # the targeted training never exercised the actual failure mode.
    key_ids = b["tokens"][0, b["key_pos"][0]]
    slots = qpos[0][err]
    weak_upd = set(key_ids[slots[slots < 500]].tolist())
    weak_keep = set(key_ids[slots[slots >= 500]].tolist())
    n_err = int(err.sum())
    return n_err, weak_upd, weak_keep, b


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
    print(f"forgetting weak-key polish from {CKPT_IN} "
          f"(scale {cfg.addr_scale:.0f}), lr {LR}", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    g = torch.Generator().manual_seed(31)
    t0 = time.time()
    probe_seed = 601
    best = 10 ** 9
    good = 0
    weak_list = []
    for s in range(MAX_STEPS):
        if s % 10 == 0:                      # refresh the weak set every 10 steps
            n_err, w_upd, w_keep, _ = census(m, probe_seed)
            probe_seed += 1
            print(f"  census {s:4d}: {n_err} errors / 1000 queries "
                  f"({len(w_upd)} weak-updated, {len(w_keep)} weak-untouched) "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            weak_upd_list = sorted(w_upd)
            weak_keep_list = sorted(w_keep)
            weak_list = weak_upd_list + weak_keep_list
            if n_err < best:
                best = n_err
                save(m, CKPT_OUT, {"census_errs": n_err})
            good = good + 1 if n_err <= 2 else 0
            if good >= 2:
                print(f"TARGET: two consecutive censuses <= 2 errors at step {s}",
                      flush=True)
                save(m, CKPT_OUT, {"census_errs": n_err})
                return
            if not weak_list:
                continue
        # targeted batch: weak keys stored, every one queried, IN ITS OWN CLASS --
        # weak-updated keys get overwritten in-batch; weak-untouched keys stay
        # untouched while other keys around them are overwritten (the failure mode)
        wk_u = torch.tensor(weak_upd_list[:32]) if weak_upd_list else torch.zeros(0, dtype=torch.long)
        wk_k = torch.tensor(weak_keep_list[:32]) if weak_keep_list else torch.zeros(0, dtype=torch.long)
        nu, nk_ = len(wk_u), len(wk_k)
        N_small = 512
        n_upd_need = N_small // 2
        # slot 0..nu-1: weak-updated keys; nu..nu+nk_-1: weak-untouched keys
        upd = torch.cat([torch.arange(nu),                                # weak-updated slots
                         (nu + nk_) + torch.randperm(N_small - nu - nk_, generator=g)
                         [: max(0, n_upd_need - nu)]]).unsqueeze(0)
        qpos = torch.cat([torch.arange(nu + nk_),                          # all weak keys
                          (nu + nk_) + torch.randperm(N_small - nu - nk_, generator=g)
                          [: 128 - nu - nk_]]).unsqueeze(0)
        wk = torch.cat([wk_u, wk_k]).long()
        if wk.numel() == 0:
            continue
        b = synth_batch(1, N_small, 128, CFG, filler=F_PROTOCOL, overwrite_frac=0.5,
                        generator=g, qpos=qpos, updated_idx=upd,
                        kw={"include_keys": wk})
        tok = b["tokens"].to(dev)
        logits = logits_at(m, tok, b["ans_pos"].to(dev))
        ce = weighted_ce(logits.reshape(-1, CFG.V), b["gold"].to(dev).reshape(-1))
        kp = torch.cat([b["key_pos"], b["upd_key_pos"]], 1)
        sl = spread_loss(m, tok, kp.to(dev), CFG, gate_subsample=512)
        loss = ce + sl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    print(f"done; best census {best} errors/1000", flush=True)


if __name__ == "__main__":
    main()
