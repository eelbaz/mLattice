#!/usr/bin/env python3
"""Capacity curriculum: 8,192 pairs at 131K tokens, target >= 0.999 (test_capacity.py).

Recipe (CONTINUE.md step 1):
  * V = 16,384 keys + 16,384 values + 2 (VS=16,384 key vocab so 8,192 stored
    keys leave half the vocab as non-key filler material);
  * addr_scale = 64  ->  (2*64+3)^2 = 17,161 cells for 8,192 keys (~2.1 keys/cell
    at the old packing, ~0.5 after the spread loss separates them);
  * kernel-D gated spread loss on the TRUE key-token addresses at every step;
  * curriculum 64 -> 256 -> 512 -> 1024 -> 2048 -> 4096 -> 8192, auto-promotion
    when held-out accuracy at the level target is reached twice consecutively;
  * warm-started from checkpoints/mqar_base.pt via vocab surgery (from-scratch
    training at V>=16K never escaped the chance plateau: logs/vocab8192.log);
  * from the 256 level on, 25% of batches carry 50% overwrites (the forgetting
    skill trained simultaneously; train_forgetting.py specialises afterwards).

The loss is cross-entropy at the query SEP positions ONLY (the head is applied
to gathered positions), so no (B,T,V) logits are ever materialised.
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

from mulattice.protocol import (SynthCfg, build_model, eval_batch, extend_from_checkpoint,
                      force_fast_scan, logits_at, spread_loss, synth_batch)

DEVICE = os.environ.get("DEVICE", "cuda")
CKPT_WARM = os.path.join(ROOT, "checkpoints", "mqar_base.pt")
CKPT_OUT = os.path.join(ROOT, "checkpoints", "capacity8k.pt")
HEARTBEAT = os.path.join(ROOT, "logs", "capacity8k.heartbeat")
HISTORY = os.path.join(ROOT, "results", "capacity_training.json")

CFG = SynthCfg()

LEVELS = [
    # pairs, max_steps, filler choices (dense-dominant), probe filler, quit acc
    dict(pairs=64,   max_steps=2500,  fillers=[0, 0, 2],    probe_f=0,  quit=0.99),
    dict(pairs=256,  max_steps=5000,  fillers=[0, 0, 2],    probe_f=0,  quit=0.99),
    dict(pairs=512,  max_steps=5000,  fillers=[0, 0, 4],    probe_f=0,  quit=0.99),
    dict(pairs=1024, max_steps=6000,  fillers=[0, 0, 4],    probe_f=0,  quit=0.99),
    dict(pairs=2048, max_steps=7000,  fillers=[0, 0, 6],    probe_f=0,  quit=0.99),
    dict(pairs=4096, max_steps=8000,  fillers=[0, 0, 6],    probe_f=0,  quit=0.99),
    dict(pairs=8192, max_steps=16000, fillers=[0, 0, 0, 4, 8], probe_f=14, quit=0.999),
]
if os.environ.get("SMOKE"):            # fast local validation of the exact code path
    LEVELS = [dict(pairs=8, max_steps=11, fillers=[0, 1], probe_f=0, quit=0.99),
              dict(pairs=16, max_steps=11, fillers=[0, 1], probe_f=1, quit=0.99)]
OVERWRITE_FRAC = 0.25      # fraction of batches (pairs>=256) that carry 50% overwrites
ALIGN_STEPS = 600          # final f=14 (131K-protocol) alignment steps at the 8192 level
if os.environ.get("SMOKE"):
    ALIGN_STEPS = 3
EVAL_EVERY = 500
if os.environ.get("SMOKE"):
    EVAL_EVERY = 5
TOKEN_BUDGET = 12288     # halved after the 175GB OOM: fewer cells per step
TOP = LEVELS[-1]["pairs"]      # the capacity target level (8,192; small in SMOKE)
START = int(os.environ.get("START_LEVEL", "0"))
if START:
    LEVELS = [lv for lv in LEVELS if lv["pairs"] >= START]
    TOP = LEVELS[-1]["pairs"]
# SCALE_END env: the top level interpolates the mixer's addr_scale from the
# checkpoint's value to SCALE_END over its first RAMP steps.  Measured
# (train_capacity4.log): at addr_scale=64, 8,192 keys pack 0.49 keys/cell and
# the level plateaus at ~0.96 (probe 0.94 at deep decay) while 4,096 keys at
# 0.24 keys/cell reach 0.98 -- density, not the mechanism, is the limiter.
# scale 96 -> (2*96+3)^2 = 38,049 cells -> 0.22 keys/cell at 8,192 keys.
SCALE_END = float(os.environ.get("SCALE_END", "0")) or None
SCALE_RAMP = int(os.environ.get("SCALE_RAMP", "3000"))

history = []
best_acc = {}
_t_save = 0.0


def beat(step, txt=""):
    with open(HEARTBEAT, "w") as f:
        f.write(f"{step} {time.time():.0f} {txt}\n")


def save(m, path, extra=None):
    live_scale = float(m.blocks[0].mixer.cfg.addr_scale)
    payload = {"model": m.state_dict(), "proto_cfg": {
        "NK": CFG.NK, "NV": CFG.NV, "d_model": CFG.d_model, "n_heads": CFG.n_heads,
        "d_k": CFG.d_k, "d_v": CFG.d_v, "addr_scale": live_scale,
        "retention_init": CFG.retention_init}, "cfg": "capacity curriculum ret9 spread"}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def held_out(m, pairs, filler, Q=64, B=2, seed=991):
    g = torch.Generator().manual_seed(seed + pairs + filler)
    b = synth_batch(B, pairs, Q, CFG, filler=filler, generator=g)
    return eval_batch(m, b, DEVICE)["acc"]


def main():
    dev = DEVICE
    torch.manual_seed(0)
    m = build_model(CFG).to(dev)
    RESUME = os.environ.get("RESUME", "")
    if RESUME and os.path.exists(RESUME):
        ck = torch.load(RESUME, map_location="cpu", weights_only=False)
        m.load_state_dict(ck["model"])
        # restore the TRAINED address scale: addr_scale is config, not weights,
        # so without this the model re-ramps 64->SCALE_END and crashes its own
        # accuracy on every restart (measured: 0.97 -> 0.34, train_capacity8.log)
        saved_scale = ck.get("proto_cfg", {}).get("addr_scale")
        if saved_scale:
            m.blocks[0].mixer.cfg.addr_scale = float(saved_scale)
        print(f"warm-start: resume from {RESUME} (step {ck.get('step')}, "
              f"scale {m.blocks[0].mixer.cfg.addr_scale:.0f})", flush=True)
    elif os.path.exists(CKPT_WARM):
        extend_from_checkpoint(m, CKPT_WARM, NK=CFG.NK, NV=CFG.NV)
        print(f"warm-start: vocab surgery from {CKPT_WARM}", flush=True)
    else:
        print("WARNING: no warm-start checkpoint; training from scratch", flush=True)
    force_fast_scan(m)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(1)
    step = 0
    t0 = time.time()
    hist_probe_best = 0.0

    def train_level(pairs, max_steps, fillers, probe_f, quit_acc, lr_sched=None):
        nonlocal step, hist_probe_best
        level_t0 = time.time()
        good = 0
        last_rounds = None
        scale0 = float(m.blocks[0].mixer.cfg.addr_scale)
        for s in range(max_steps):
            if SCALE_END and pairs <= TOP:
                frac = min(1.0, s / SCALE_RAMP)
                m.blocks[0].mixer.cfg.addr_scale = scale0 + (SCALE_END - scale0) * frac
            f = fillers[step % len(fillers)]
            # Q<=16 in TRAINING: the fastscan autograd graph holds ~rounds x n_cells
            # cell copies; at Q=64 (rounds~67) x 179K cells (stretched grid) that is
            # ~175GB total-vm and the host OOM-killer fires (measured, exit 137,
            # train_capacity6.log).  Large-Q accuracy is measured by no-grad probes.
            # Memory-aware Q: at B<=2 (pairs>=2048) cells<=40K so Q=64 costs
            # ~20GB (safe) and is NEEDED -- with Q<=16 the CE signal collapses
            # and the big levels erode (p2048/p4096 declined to 0.88-0.93,
            # train_capacity7.log).  Small levels keep B>1 -> Q<=16 for memory.
            Q = (64 if step % 4 == 0 else 16) if pairs >= 2048 else (8 if step % 4 else 16)
            ow = OVERWRITE_FRAC if (pairs >= 256 and step % 4 == 1) else 0.0
            T_est = int(pairs * (1 + ow) * (f + 2)) + 2 * Q
            B = max(1, min(32, TOKEN_BUDGET // T_est))
            b = synth_batch(B, pairs, Q, CFG, filler=f, overwrite_frac=ow, generator=g)
            tok = b["tokens"].to(dev)
            ans = b["ans_pos"].to(dev)
            logits = logits_at(m, tok, ans)
            ce = F.cross_entropy(logits.reshape(-1, CFG.V), b["gold"].to(dev).reshape(-1))
            sl = spread_loss(m, tok, b["key_pos"].to(dev), CFG)
            loss = ce + sl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            if lr_sched:
                lr_sched.step()
            step += 1
            if s % 100 == 0 or s == max_steps - 1:
                sc = m.blocks[0].mixer.scan
                print(f"  p{pairs:5d} f{f:2d} step {step:6d} ce {float(ce.detach()):.3f} "
                      f"spread {float(sl.detach()):.3f} rounds {sc.last_rounds} cells {sc.last_n_cells} "
                      f"B{B} T{b['L']} {time.time()-t0:6.0f}s", flush=True)
                beat(step, f"pairs{pairs}")
            if (s + 1) % EVAL_EVERY == 0 or s == max_steps - 1:
                acc = held_out(m, pairs, fillers[-1], Q=128)
                # per-level rolling checkpoints: latest AND best-by-acc, so a
                # degraded level tail can never clobber its own best model
                save(m, os.path.join(ROOT, "checkpoints", f"capacity8k_p{pairs}_latest.pt"),
                     {"step": step, "acc": acc})
                if acc >= best_acc.get(pairs, 0.0):
                    best_acc[pairs] = acc
                    save(m, os.path.join(ROOT, "checkpoints", f"capacity8k_p{pairs}_best.pt"),
                         {"step": step, "acc": acc})
                rec = {"step": step, "pairs": pairs, "acc": round(acc, 4), "t": round(time.time() - t0)}
                probe = None
                if pairs == 8192 and probe_f > 0:
                    probe = held_out(m, pairs, probe_f, Q=512, B=1, seed=5)
                    rec["probe_131k"] = round(probe, 4)
                    hist_probe_best = max(hist_probe_best, probe)
                history.append(rec)
                with open(HISTORY, "w") as fh:
                    json.dump(history, fh, indent=2)
                print(f"  EVAL p{pairs} f{fillers[-1]} acc {acc:.4f} sc{float(m.blocks[0].mixer.cfg.addr_scale):.0f}"
                      + (f" probe@f{probe_f} {probe:.4f}" if probe is not None else "")
                      + f"  [{time.time()-level_t0:.0f}s in level]", flush=True)
                good = good + 1 if acc >= quit_acc else 0
                if good >= 2 and pairs < TOP:
                    print(f"  PROMOTE at step {step} (acc {acc:.4f} x2 >= {quit_acc})", flush=True)
                    break
                if pairs == TOP and probe is not None and probe >= 0.999 and acc >= 0.999:
                    print(f"  TARGET REACHED at step {step}: dense {acc:.4f}, 131K probe {probe:.4f}",
                          flush=True)
                    return True
        return False

    print(f"CAPACITY CURRICULUM  V={CFG.V} addr_scale={CFG.addr_scale} device={dev}", flush=True)
    for i, lv in enumerate(LEVELS):
        # PER-LEVEL cosine decay.  Measured (train_capacity3.log): with constant
        # lr 3e-3 the p512 level oscillated around 0.985 held-out for its whole
        # 5,000-step budget with no upward trend (training ce stuck at 0.03-0.05)
        # -- the level never squeezes its last errors.  The realdata run only
        # reached 1.0000 once cosine decay started.  Each level now decays
        # 3e-3 -> 1e-3 (top level: -> 3e-4 over steps+align).
        # Level-scaled entry LR + warmup.  A fresh 3e-3 restart at every level
        # EXPLODES a converged model at the big levels (B=1-2, ce 0.085 -> 5.4 in
        # 100 steps, train_capacity9.log); small levels with B>=5 survive it.
        # Entry: 3e-3 (<=1024), 1.5e-3 (2048), 1e-3 (>=4096); 150-step warmup,
        # cosine to eta_min over the level (TOP: 3e-4, else entry/3).
        import math as _math
        lr0 = 3e-3 if lv["pairs"] <= 1024 else (1.5e-3 if lv["pairs"] == 2048 else 1e-3)
        eta = 3e-4 if lv["pairs"] == TOP else lr0 / 3.0
        total = lv["max_steps"] + (ALIGN_STEPS if lv["pairs"] == TOP else 0)
        warm = 150
        fmin = eta / lr0

        def _lam(ss):
            if ss < warm:
                return (ss + 1) / warm
            prog = min(1.0, (ss - warm) / max(1, total - warm))
            return fmin + (1 - fmin) * 0.5 * (1 + _math.cos(_math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr0 * (1.0 / warm)
        lr_sched = torch.optim.lr_scheduler.LambdaLR(opt, _lam)
        done = train_level(lv["pairs"], lv["max_steps"], lv["fillers"], lv["probe_f"], lv["quit"],
                           lr_sched)
        save(m, os.path.join(ROOT, "checkpoints", f"capacity8k_lvl{lv['pairs']}.pt"),
             {"step": step})
        if done:
            break

    # final alignment: the 131K filler distribution at memory-safe query counts.
    # (Q>=512 training steps OOM: rounds~Q cell-copies in the autograd graph --
    # the realdata align died of exactly this.  Large-Q accuracy is measured by
    # the no-grad probes and the acceptance test, never trained.)
    print(f"ALIGN phase: f=14 (131K protocol), {ALIGN_STEPS} steps, Q<=64", flush=True)
    for s in range(ALIGN_STEPS):
        Q = 64 if s % 3 == 0 else 16
        b = synth_batch(1, TOP, Q, CFG, filler=14, overwrite_frac=0.25 if s % 4 == 1 else 0.0,
                        generator=g)
        tok = b["tokens"].to(dev)
        logits = logits_at(m, tok, b["ans_pos"].to(dev))
        ce = F.cross_entropy(logits.reshape(-1, CFG.V), b["gold"].to(dev).reshape(-1))
        sl = spread_loss(m, tok, b["key_pos"].to(dev), CFG, gate_subsample=1024)
        loss = ce + sl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        step += 1
        if s % 50 == 0 or s == ALIGN_STEPS - 1:
            print(f"  align {s:4d} step {step} ce {float(ce.detach()):.3f} T{b['L']} "
                  f"{time.time()-t0:6.0f}s", flush=True)
            beat(step, "align")
            acc = held_out(m, TOP, 14, Q=512, B=1, seed=5)
            save(m, os.path.join(ROOT, "checkpoints", "capacity8k_latest.pt"), {"step": step})
            history.append({"step": step, "align": True, "probe_131k": round(acc, 4)})
            with open(HISTORY, "w") as fh:
                json.dump(history, fh, indent=2)
            print(f"  ALIGN probe@131K acc {acc:.4f}", flush=True)

    save(m, CKPT_OUT, {"step": step})
    print(f"saved {CKPT_OUT} at step {step}, total {time.time()-t0:.0f}s", flush=True)
    # final self-check with the test protocol
    a1 = held_out(m, TOP, 14, Q=1024, B=1, seed=20261102)
    a2 = held_out(m, TOP, 0, Q=1024, B=1, seed=20261103)
    print(f"SELF-CHECK  131K protocol {a1:.4f}   dense {a2:.4f}", flush=True)
    history.append({"final": True, "acc_131k": round(a1, 4), "acc_dense": round(a2, 4)})
    with open(HISTORY, "w") as fh:
        json.dump(history, fh, indent=2)


if __name__ == "__main__":
    main()
