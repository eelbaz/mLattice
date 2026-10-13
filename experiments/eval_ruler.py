#!/usr/bin/env python3
"""RULER-style single-needle retrieval evaluation.

RULER S-NIAH (Single Needle In A Haystack) is the standard long-context
benchmark: embed one fact ("needle") in a long context of distractor text
("haystack"), then ask the model to retrieve it. This tests whether the mixer
can find one specific association among many tokens of noise.

Protocol:
  * Generate a haystack of N key-value pairs (distractors) from the vocabulary
  * Embed ONE target pair (the needle) at a specified depth
  * After the full sequence, query the needle's key
  * The model must return the needle's value

This differs from capacity (which queries many pairs) — here we test whether
a SINGLE fact survives among noise, which is the harder retrieval problem.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from mulattice.protocol import SynthCfg, build_model, force_fast_scan, logits_at, synth_batch

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
CKPT = os.path.join(ROOT, "checkpoints", "capacity8k.pt")


def run_needle(n_haystack: int, filler: int, depth_frac: float, seed: int, cfg, model):
    """Place one needle at `depth_frac` through a haystack of n_haystack pairs."""
    g = torch.Generator().manual_seed(seed)

    # Build a standard batch, then extract the needle at the right position
    b = synth_batch(1, n_haystack + 1, 1, cfg, filler=filler, generator=g)
    tokens = b["tokens"]
    L = tokens.shape[1]

    # The needle is the pair at index `needle_idx` in the store pass
    block = filler + 2
    needle_idx = int(n_haystack * depth_frac)
    needle_key = int(tokens[0, needle_idx * block + filler])
    needle_val = int(tokens[0, needle_idx * block + filler + 1])

    # Query the needle
    S2 = (n_haystack + 1) * block
    tokens[0, S2] = needle_key
    tokens[0, S2 + 1] = cfg.SEP
    ans_pos = torch.tensor([[S2 + 1]])
    gold = torch.tensor([[needle_val]])

    with torch.no_grad():
        pred = logits_at(model, tokens.to(DEVICE), ans_pos.to(DEVICE)).argmax(-1).cpu()
    correct = int(pred[0, 0]) == int(gold[0, 0])
    return correct, needle_key, needle_val, int(pred[0, 0]), L


def main():
    cfg = SynthCfg()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg.addr_scale = float(ck["proto_cfg"]["addr_scale"])
    model = build_model(cfg)
    model.load_state_dict(ck["model"])
    model = model.to(DEVICE).eval()
    force_fast_scan(model)

    print(f"RULER S-NIAH evaluation on {DEVICE}", flush=True)
    print(f"model: capacity8k.pt, scale={cfg.addr_scale:.0f}", flush=True)
    print()

    # Test at increasing haystack sizes and depths
    configs = []
    for n_hay in [256, 1024, 4096, 8192]:
        for depth in [0.05, 0.25, 0.50, 0.75, 0.95]:
            configs.append((n_hay, 14, depth))

    results = []
    n_correct = 0
    n_total = 0

    for n_hay, filler, depth in configs:
        # 5 seeds per config
        for seed in range(5):
            ok, k, v, p, L = run_needle(n_hay, filler, depth, seed + 100, cfg, model)
            results.append({
                "n_haystack": n_hay, "filler": filler, "depth": depth,
                "seed": seed, "correct": bool(ok), "T": int(L),
            })
            n_correct += ok
            n_total += 1

        acc = sum(r["correct"] for r in results if r["n_haystack"] == n_hay) / \
              sum(1 for r in results if r["n_haystack"] == n_hay)
        T = n_hay * 16 + 2
        print(f"  haystack={n_hay:5d} pairs, T≈{T:6d}: "
              f"{sum(r['correct'] for r in results if r['n_haystack']==n_hay)}/"
              f"{sum(1 for r in results if r['n_haystack']==n_hay)} "
              f"({acc:.4f})", flush=True)

    overall = n_correct / n_total
    print(f"\nOverall: {n_correct}/{n_total} = {overall:.4f}", flush=True)

    # Breakdown by depth
    print("\nBy depth:")
    for d in [0.05, 0.25, 0.50, 0.75, 0.95]:
        sub = [r for r in results if r["depth"] == d]
        acc = sum(r["correct"] for r in sub) / len(sub)
        print(f"  depth {d:4.0%}: {acc:.4f} ({sum(r['correct'] for r in sub)}/{len(sub)})")

    out = os.path.join(ROOT, "results", "ruler_sniah.json")
    with open(out, "w") as f:
        json.dump({"overall": overall, "results": results}, f, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
