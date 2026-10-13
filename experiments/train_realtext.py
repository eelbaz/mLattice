#!/usr/bin/env python3
"""Real-data 259K retrieval at the 0.999 level (test_realtext.py).

v3 design (after two measured failures, documented below):

  * d_model=64 -- the PROVEN architecture.  A d_model=128 rebuild failed to
    transition twice (uniform keys: frozen at h@64 0.3516 for 3.4K steps; the
    faithful Zipf recipe: frozen at 0.2422 for 3.6K steps -- while the d=64
    original transitioned in <400 steps).  Width changes re-roll the
    transition dice; we don't take that gamble.
  * warm-started from checkpoints/realtext_base.pt (our own 0.875 model) -- no
    transition needed at all; this run repairs and extends it.
  * kernel-D spread loss at the TRUE key-token positions (the 0.875 run applied
    it at `0::2` of the whole sequence -- mostly filler -- so it never acted).
  * growth to the full 3,548-pair protocol (the old run grew only to 512 pairs,
    then evaluated at 3,548 -- out of distribution).
  * the address scale stays at 181 (the checkpoint's final scale; its
    degradation came from hitting 181 in the last ~150 steps of training, not
    from the scale itself -- we re-stabilise at 181 for thousands of steps).
  * long cosine polish at the full pair count, then an alignment phase at the
    exact eval distribution (filler=70, Q=2,048, depth-stratified, T~=259K).

Protocol identical to experiments/eval_realdata.py: wikitext-103 slice,
VS=4,094 word vocab, key->bigram-successor pairs (count>=4), real-text filler.
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

from mulattice.protocol import RealData, logits_at, stratified_qpos
from mulattice.models import MQARModel, MixerCfg

DEVICE = os.environ.get("DEVICE", "cuda")
CORPUS = os.path.join(ROOT, "data", "wikitext103_slice.txt")
CKPT_WARM = os.environ.get("WARM", os.path.join(ROOT, "checkpoints", "realtext_base.pt"))
CKPT_OUT = os.path.join(ROOT, "checkpoints", "realtext.pt")
HEARTBEAT = os.path.join(ROOT, "logs", "realtext.heartbeat")
HISTORY = os.path.join(ROOT, "results", "realtext_training.json")

D_MODEL = 64
SCALE = 181.0
P_EVAL = 3548          # protocol pair cap (mining yields 3,804; eval_realdata caps at 3,548)
F_EVAL = 70            # 3548*72 + 2*2048 = 259,552 ~= 259K
Q_EVAL = 2048
TOKEN_BUDGET = 16384

RESTAB = [(64, 400), (128, 400), (256, 800)]          # re-stabilise the warm model
GROW = [(512, 1500), (1024, 2000), (2048, 2500), (3548, 2500)]
POLISH_STEPS = 1500
ALIGN_STEPS = 60
if os.environ.get("SMOKE"):            # fast local validation of the exact code path
    RESTAB = [(8, 3), (16, 3)]
    GROW = [(32, 3), (64, 3)]
    POLISH_STEPS = 3
    ALIGN_STEPS = 2


def beat(step, txt=""):
    with open(HEARTBEAT, "w") as f:
        f.write(f"{step} {txt}\n")


def spread_loss_real(m, tokens, key_pos, scale, gate_subsample=2048):
    from mulattice.protocol import _gated_spread
    mx = m.blocks[0].mixer
    xc = mx.conv(m.blocks[0].n1(m.emb(tokens)))
    kaddr = scale * torch.tanh(mx.Wa(xc))
    ar = torch.arange(tokens.shape[0], device=tokens.device).unsqueeze(1)
    addr = kaddr[ar, key_pos]
    B, N, _ = addr.shape
    if N > gate_subsample:
        addr = addr[:, torch.randperm(N, device=addr.device)[:gate_subsample]]
    return _gated_spread(addr, addr, scale)


def main():
    rd = RealData(CORPUS)
    V = rd.V
    print(f"corpus: {rd.n_pairs} kv-pairs (cap {P_EVAL}), filler stream {len(rd.FILLER)}",
          flush=True)
    dev = DEVICE
    torch.manual_seed(0)
    m = MQARModel(V, kind="mulattice", n_layers=1,
                  cfg=MixerCfg(d_model=D_MODEL, n_heads=2, d_k=D_MODEL // 2, d_v=D_MODEL // 2,
                               addr_scale=SCALE, learn_addr_scale=False,
                               bound_address=True, retention_init=9.0)).to(dev)
    if os.path.exists(CKPT_WARM):
        ck = torch.load(CKPT_WARM, map_location="cpu", weights_only=False)
        m.load_state_dict(ck["model"])
        print(f"warm-start from {CKPT_WARM}", flush=True)
    else:
        print(f"WARNING: {CKPT_WARM} missing, training from scratch", flush=True)
    mx = m.blocks[0].mixer
    mx.scan.crossover = 1e9            # cell-parallel from the start (T is large)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(1)
    step = 0
    t0 = time.time()
    history = []

    def held_out(pairs, filler, Q=64, B=2, seed=991, zipf=False):
        m.eval()
        c = t = 0
        with torch.no_grad():
            for _ in range(B):
                b = rd.batch(pairs, filler, Q, seed=seed + _, zipf=zipf)
                pred = logits_at(m, b["tokens"].to(dev), b["ans_pos"].to(dev)).argmax(-1).cpu()
                c += int((pred == b["gold"]).sum()); t += pred.numel()
        m.train()
        return c / t

    def ce_step(pairs, filler, Q, lr_decay=None, zipf=False, use_spread=True):
        nonlocal step
        T_est = pairs * (filler + 2) + 2 * Q
        B = max(1, min(8, TOKEN_BUDGET // T_est))
        b = rd.batch(pairs, filler, Q, seed=int(g.initial_seed()) + step, zipf=zipf)
        tok = b["tokens"].to(dev)
        logits = logits_at(m, tok, b["ans_pos"].to(dev))
        ce = F.cross_entropy(logits.reshape(-1, V), b["gold"].to(dev).reshape(-1))
        if use_spread:
            sl = spread_loss_real(m, tok, b["key_pos"].to(dev), SCALE)
            loss = ce + sl
        else:
            sl = 0.0
            loss = ce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if lr_decay:
            lr_decay.step()
        step += 1
        return float(ce.detach()), (float(sl) if use_spread else 0.0), b["L"], B

    # ---------------- re-stabilise the warm model (its last run ended degraded).
    # f=2: light filler, close to the warm model's original training distribution.
    print(f"RESTAB (pairs,steps): {RESTAB}  zipf, f=2, no spread, lr 1e-3", flush=True)
    for pairs, n in RESTAB:
        for s in range(n):
            ce, sl, L, B = ce_step(pairs, 2, 4, zipf=True, use_spread=False)
            if s % 200 == 0 or s == n - 1:
                a = held_out(64, 2, zipf=True)
                print(f"  r p{pairs:4d} step {step:6d} ce {ce:.3f} h@64 {a:.4f} "
                      f"{time.time()-t0:5.0f}s", flush=True)
                beat(step, f"restab p{pairs}")
                history.append({"step": step, "phase": "restab", "pairs": pairs,
                                "h64": round(a, 4)})

    # ---------------- growth to the full protocol pair count, true-key spread
    print(f"GROWTH (pairs,steps): {GROW}  zipf/uniform alternation + true-key spread",
          flush=True)
    for pairs, n in GROW:
        good = 0
        for s in range(n):
            ce, sl, L, B = ce_step(pairs, 2, 8, zipf=(step % 2 == 0), use_spread=True)
            if s % 200 == 0 or s == n - 1:
                a = held_out(pairs, 2)
                a64 = held_out(64, 0, zipf=True)
                print(f"  g p{pairs:4d} step {step:6d} ce {ce:.3f} spread {sl:.3f} "
                      f"h@{pairs} {a:.4f} h@64 {a64:.4f} rounds {mx.scan.last_rounds} "
                      f"{time.time()-t0:5.0f}s", flush=True)
                beat(step, f"grow p{pairs}")
                history.append({"step": step, "phase": "grow", "pairs": pairs,
                                "acc": round(a, 4), "h64": round(a64, 4)})
                with open(HISTORY, "w") as fh:
                    json.dump(history, fh, indent=2)
                torch.save({"model": m.state_dict(),
                            "proto_cfg": {"V": V, "d_model": D_MODEL, "addr_scale": SCALE,
                                          "n_heads": 2, "d_k": D_MODEL // 2, "d_v": D_MODEL // 2},
                            "cfg": "realtext periodic", "step": step},
                           os.path.join(ROOT, "checkpoints", "realtext_latest.pt"))
                good = good + 1 if a >= 0.99 else 0
                if good >= 2 and pairs < P_EVAL:
                    print(f"  PROMOTE p{pairs} at step {step}", flush=True)
                    break

    # ---------------- polish: full protocol pair count, cosine decay, uniform keys
    print(f"POLISH {POLISH_STEPS} steps at P={P_EVAL}, filler cycle [2,2,4,8], cosine 1e-3->1e-4",
          flush=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=POLISH_STEPS + ALIGN_STEPS,
                                                       eta_min=1e-4)
    fcyc = [2, 2, 4, 8]
    for s in range(POLISH_STEPS):
        ce, sl, L, B = ce_step(P_EVAL, fcyc[s % len(fcyc)], 16, lr_decay=sched,
                               zipf=False, use_spread=True)
        if s % 100 == 0 or s == POLISH_STEPS - 1:
            print(f"  polish {s:4d} step {step} ce {ce:.3f} T{L} {time.time()-t0:6.0f}s", flush=True)
            beat(step, "polish")
        if (s + 1) % 250 == 0:
            a = held_out(P_EVAL, 8, Q=128, B=1)
            print(f"  POLISH EVAL acc {a:.4f}", flush=True)
            history.append({"step": step, "phase": "polish", "acc": round(a, 4)})
            with open(HISTORY, "w") as fh:
                json.dump(history, fh, indent=2)
            torch.save({"model": m.state_dict(),
                        "proto_cfg": {"V": V, "d_model": D_MODEL, "addr_scale": SCALE,
                                      "n_heads": 2, "d_k": D_MODEL // 2, "d_v": D_MODEL // 2},
                        "cfg": "realtext periodic", "step": step},
                       os.path.join(ROOT, "checkpoints", "realtext_latest.pt"))

    # ---------------- align: exact eval distribution (filler=70, Q=2048, stratified)
    print(f"ALIGN {ALIGN_STEPS} steps at filler={F_EVAL} (Q=64 train-safe; "
          f"Q={Q_EVAL} only under no_grad)", flush=True)
    qpos = stratified_qpos(P_EVAL, 64, 4, seed=4242)
    for s in range(ALIGN_STEPS):
        b = rd.batch(P_EVAL, F_EVAL, 64, seed=777000 + s, qpos_idx=qpos)
        tok = b["tokens"].to(dev)
        logits = logits_at(m, tok, b["ans_pos"].to(dev))
        ce = F.cross_entropy(logits.reshape(-1, V), b["gold"].to(dev).reshape(-1))
        sl = spread_loss_real(m, tok, b["key_pos"].to(dev), SCALE, gate_subsample=1024)
        loss = ce + sl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if s % 10 == 0 or s == ALIGN_STEPS - 1:
            torch.save({"model": m.state_dict(),
                        "proto_cfg": {"V": V, "d_model": D_MODEL, "addr_scale": SCALE,
                                      "n_heads": 2, "d_k": D_MODEL // 2, "d_v": D_MODEL // 2},
                        "cfg": "realtext periodic", "step": step},
                       os.path.join(ROOT, "checkpoints", "realtext_latest.pt"))
            print(f"  align {s:3d} step {step} ce {ce:.3f} T{b['L']} {time.time()-t0:6.0f}s",
                  flush=True)
            beat(step, "align")

    torch.save({"model": m.state_dict(),
                "proto_cfg": {"V": V, "d_model": D_MODEL, "addr_scale": SCALE,
                              "n_heads": 2, "d_k": D_MODEL // 2, "d_v": D_MODEL // 2},
                "cfg": "realdata_v2 d64 warm256k spread-true-keys growth3548 polish align"},
               CKPT_OUT)
    print(f"saved {CKPT_OUT}", flush=True)

    # ---------------- final self-check: the acceptance protocol, both seeds
    m.eval()
    for seed in (4242, 4243):
        qp = stratified_qpos(P_EVAL, Q_EVAL, 4, seed=seed)
        b = rd.batch(P_EVAL, F_EVAL, Q_EVAL, seed=seed, qpos_idx=qp)
        with torch.no_grad():
            pred = logits_at(m, b["tokens"].to(dev), b["ans_pos"].to(dev)).argmax(-1).cpu()
        ok = (pred == b["gold"]).squeeze(0)
        per = Q_EVAL // 4
        strata = [round(ok[i * per:(i + 1) * per].float().mean().item(), 4) for i in range(4)]
        overall = ok.float().mean().item()
        print(f"SELF-CHECK seed {seed}: overall {overall:.4f} strata {strata} T={b['L']}",
              flush=True)
        history.append({"step": step, "final": True, "seed": seed,
                        "overall": round(overall, 4), "strata": strata})
    with open(HISTORY, "w") as fh:
        json.dump(history, fh, indent=2)


if __name__ == "__main__":
    main()
