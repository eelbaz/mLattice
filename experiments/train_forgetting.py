#!/usr/bin/env python3
"""Forgetting training: after overwrite, queries must return the UPDATED value.

Protocol (CONTINUE.md step 2): sequences where 50% of keys appear twice (first
the original value, later an updated value).  The loss is on the query positions
AFTER the updates; the delta rule must learn that the LAST value wins.

Warm-started from checkpoints/capacity8k.pt (the capacity curriculum already
mixes 25% overwrite batches, so this run specialises an existing skill rather
than bootstrapping one).

Curriculum: 64 -> 256 -> 1024 -> 4096 pairs, overwrite fraction cycling through
{0.3, 0.5, 0.7}, both dense (f=0) and 131K-protocol (f=14) batches.  Final phase
trains the exact acceptance protocol (N=5,461, f=14, 50% overwrite) and quits
when the held-out probe (500 updated + 500 untouched queries) is >= 0.998 on
both query classes.
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
CFG = SynthCfg()
CKPT_WARM = os.environ.get("WARM", os.path.join(ROOT, "checkpoints", "capacity8k.pt"))
CKPT_OUT = os.path.join(ROOT, "checkpoints", "forgetting.pt")
HEARTBEAT = os.path.join(ROOT, "logs", "forgetting.heartbeat")
HISTORY = os.path.join(ROOT, "results", "forgetting_training.json")

LEVELS = [
    dict(pairs=64,   steps=1500),
    dict(pairs=256,  steps=2000),
    dict(pairs=1024, steps=2500),
    dict(pairs=4096, steps=4000),
]
FINAL = dict(pairs=5461, steps=6000, filler=14)   # the acceptance protocol
OW_CYCLE = [0.3, 0.5, 0.7, 0.5]
TOKEN_BUDGET = 20000
if os.environ.get("SMOKE"):            # fast local validation of the exact code path
    LEVELS = [dict(pairs=8, steps=5), dict(pairs=16, steps=5)]
    FINAL = dict(pairs=24, steps=6, filler=3)


def save(m, path, extra=None):
    # record the LIVE mixer scale (the FORK ramp mutates it; writing the stale
    # CFG value made scale-128 weights masquerade as scale 96 on reload)
    live = float(m.blocks[0].mixer.cfg.addr_scale)
    payload = {"model": m.state_dict(), "proto_cfg": {
        "NK": CFG.NK, "NV": CFG.NV, "d_model": CFG.d_model, "n_heads": CFG.n_heads,
        "d_k": CFG.d_k, "d_v": CFG.d_v, "addr_scale": live,
        "retention_init": CFG.retention_init}, "cfg": "forgetting overwrite curriculum"}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def held_out(m, pairs, filler, n_u=200, n_k=200, seed=777):
    """Acceptance-shaped probe: half queries on updated, half on untouched keys.

    Query counts are clamped to the pool sizes (n_u <= n_upd, n_k <= n_pairs-n_upd)."""
    g = torch.Generator().manual_seed(seed + pairs + filler)
    n_upd = pairs // 2
    perm = torch.randperm(pairs, generator=g)
    upd_idx = perm[:n_upd]
    upd_set = set(upd_idx.tolist())
    updated_pool = [i for i in perm.tolist() if i in upd_set]
    untouched_pool = [i for i in perm.tolist() if i not in upd_set]
    qpos = torch.tensor(updated_pool[:min(n_u, len(updated_pool))]
                        + untouched_pool[:min(n_k, len(untouched_pool))]).unsqueeze(0)
    b = synth_batch(1, pairs, qpos.shape[1], CFG, filler=filler, overwrite_frac=0.5,
                    generator=g, qpos=qpos, updated_idx=upd_idx.unsqueeze(0))
    return eval_batch(m, b, DEVICE)


def main():
    dev = DEVICE
    torch.manual_seed(0)
    m = build_model(CFG).to(dev)
    warm = CKPT_WARM if os.path.exists(CKPT_WARM) else os.path.join(
        ROOT, "checkpoints", "capacity8k_latest.pt")
    if os.path.exists(warm):
        ck = torch.load(warm, map_location="cpu", weights_only=False)
        m.load_state_dict(ck["model"])
        # restore the TRAINED grid scale (addr_scale is config, not weights).
        # Without this the forgetting lineage ran at scale 64 while its
        # capacity ancestor was trained at 96. At N=5,461 + 50% overwrites the
        # sequence stores 8,192 pair-blocks; at scale 64 this exceeds the
        # density ceiling (see LIMITATIONS.md).
        saved_scale = ck.get("proto_cfg", {}).get("addr_scale")
        if saved_scale:
            m.blocks[0].mixer.cfg.addr_scale = float(saved_scale)
            CFG.addr_scale = float(saved_scale)
        print(f"warm-start from {warm} (scale {m.blocks[0].mixer.cfg.addr_scale:.0f})",
              flush=True)
    else:
        print("WARNING: no warm-start checkpoint at all, training from scratch", flush=True)
    force_fast_scan(m)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=sum(l["steps"] for l in LEVELS) + FINAL["steps"], eta_min=1e-4)
    g = torch.Generator().manual_seed(2)
    step = 0
    t0 = time.time()
    history = []

    def train_steps(pairs, n_steps, filler_cycle, probe_f=None):
        nonlocal step
        good = 0
        for s in range(n_steps):
            ow = OW_CYCLE[step % len(OW_CYCLE)]
            f = filler_cycle[step % len(filler_cycle)]
            Q = 8 if step % 4 else 64
            T_est = int(pairs * (1 + ow) * (f + 2)) + 2 * Q
            B = max(1, min(16, TOKEN_BUDGET // T_est))
            b = synth_batch(B, pairs, Q, CFG, filler=f, overwrite_frac=ow, generator=g)
            tok = b["tokens"].to(dev)
            logits = logits_at(m, tok, b["ans_pos"].to(dev))
            ce = F.cross_entropy(logits.reshape(-1, CFG.V), b["gold"].to(dev).reshape(-1))
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
            step += 1
            if s % 100 == 0 or s == n_steps - 1:
                sc = m.blocks[0].mixer.scan
                print(f"  p{pairs:5d} f{f:2d} ow{ow:.1f} step {step:6d} ce {float(ce.detach()):.3f} "
                      f"spread {float(sl.detach()):.3f} rounds {sc.last_rounds} "
                      f"{time.time()-t0:6.0f}s", flush=True)
                with open(HEARTBEAT, "w") as fh:
                    fh.write(f"{step}\n")
            if (s + 1) % 500 == 0 or s == n_steps - 1:
                r = held_out(m, pairs, filler_cycle[-1])
                history.append({"step": step, "pairs": pairs, "f": filler_cycle[-1],
                                "acc_updated": round(r["acc_updated"], 4),
                                "acc_untouched": round(r["acc_untouched"], 4)})
                with open(HISTORY, "w") as fh:
                    json.dump(history, fh, indent=2)
                print(f"  EVAL p{pairs} upd {r['acc_updated']:.4f} keep {r['acc_untouched']:.4f}",
                      flush=True)

    print(f"FORGETTING CURRICULUM warm={warm} device={dev}", flush=True)
    START = int(os.environ.get("START_LEVEL", "0"))
    ladder = [lv for lv in LEVELS if lv["pairs"] >= START]
    for lv in ladder:
        train_steps(lv["pairs"], lv["steps"], [0, 0, 14])
        save(m, os.path.join(ROOT, "checkpoints", f"forgetting_lvl{lv['pairs']}.pt"), {"step": step})

    # FINAL phase: a LADDER, not a jump.  Measured: the overwrite skill scales
    # down with pair count (1.0000 @64/256, ~0.98 @1024, ~0.95 @4096) exactly
    # like capacity did before its ladder; and f=14 steps cost ~10s (T=131K) so
    # a single-N final is both too shallow and too slow.  Train at f in {6,8,14}
    # (the mechanism is filler-agnostic; the acceptance distribution gets its
    # own align steps), N ladder 2048->3548->5461, probe on the exact protocol.
    # FORK envs (fork-and-flush, cf. arXiv:2610.07447): a parallel trajectory
    # with DIFFERENT basin assumptions -- a bigger grid for double-writes
    # (FORK_SCALE ramps during the final ladder) and long-gap erasure training
    # (FORK_FILLERS raises the filler mix, stretching store->update gaps).
    FS = float(os.environ.get("FORK_SCALE", "0")) or None
    FF = os.environ.get("FORK_FILLERS", "6,8,14,8")
    fcyc_fork = [int(x) for x in FF.split(",")]
    print(f"FINAL phase: ladder 2048->3548->5461, filler mix {fcyc_fork}"
          + (f", scale ramp ->{FS:.0f}" if FS else ""), flush=True)
    FO = os.environ.get("FINAL_ONLY", "")
    if FO:
        FINAL_LADDER = [(5461, int(FO))]
    else:
        FINAL_LADDER = [(2048, 1200), (3548, 1500), (5461, 3000)]
    scale0_f = float(m.blocks[0].mixer.cfg.addr_scale)
    LR0 = float(os.environ.get("FINAL_LR", "0"))   # entry LR for the final
    # ladder; without it the phase runs at the optimizer's construction LR
    # (1e-3), which EXPLODES a converged model at B=1 (measured: probes
    # 0.988 -> 0.754 in 200 steps -- Law 3 again).  Warm up over 100 steps.
    WARM = 100
    import math as _m
    for FL_N, FL_STEPS in FINAL_LADDER:
        for s in range(FL_STEPS):
            if LR0:
                frac = min(1.0, (s + 1) / WARM)
                cur = LR0 * (0.05 + 0.95 * frac)
                tail = LR0 * (1.0 - 0.75 * min(1.0, s / max(1, FL_STEPS - 1)))
                lr = min(cur, tail) if s >= WARM else cur
                for pg in opt.param_groups:
                    pg["lr"] = lr
            if FS:
                frac = min(1.0, step / 2000.0)
                m.blocks[0].mixer.cfg.addr_scale = scale0_f + (FS - scale0_f) * frac
            ow = OW_CYCLE[s % len(OW_CYCLE)]
            f = fcyc_fork[s % len(fcyc_fork)]
            b = synth_batch(1, FL_N, 64, CFG, filler=f,
                            overwrite_frac=ow, generator=g)
            tok = b["tokens"].to(dev)
            logits = logits_at(m, tok, b["ans_pos"].to(dev))
            ce = F.cross_entropy(logits.reshape(-1, CFG.V), b["gold"].to(dev).reshape(-1))
            kp = torch.cat([b["key_pos"], b["upd_key_pos"]], 1)
            sl = spread_loss(m, tok, kp.to(dev), CFG, gate_subsample=1024)
            loss = ce + sl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if s % 100 == 0 or s == FL_STEPS - 1:
                print(f"  final N{FL_N} f{f} {s:4d} ce {float(ce.detach()):.3f} T{b['L']} "
                      f"{time.time()-t0:6.0f}s", flush=True)
            if (s + 1) % 200 == 0 or s == FL_STEPS - 1:
                r = held_out(m, 5461, 14, n_u=500, n_k=500, seed=777001)
                history.append({"step": step, "final": FINAL["pairs"] if False else FL_N,
                                "acc_updated": round(r["acc_updated"], 4),
                                "acc_untouched": round(r["acc_untouched"], 4)})
                with open(HISTORY, "w") as fh:
                    json.dump(history, fh, indent=2)
                print(f"  FINAL PROBE upd {r['acc_updated']:.4f} keep {r['acc_untouched']:.4f}",
                      flush=True)
                save(m, CKPT_OUT, {"step": step})
                if r["acc_updated"] >= 0.998 and r["acc_untouched"] >= 0.998:
                    print(f"TARGET REACHED step {step}", flush=True)
                    break
        else:
            continue
        break  # TARGET REACHED inside the inner loop -> stop the ladder too
    save(m, CKPT_OUT, {"step": step})
    r = held_out(m, 64, 0, n_u=256, n_k=256, seed=777003)
    print(f"SELF-CHECK dense N=64: upd {r['acc_updated']:.4f} keep {r['acc_untouched']:.4f}",
          flush=True)
    print(f"saved {CKPT_OUT} total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
