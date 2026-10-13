#!/usr/bin/env python3
"""Characterize addressing limits: adversarial keys, packing, collision scaling.

Tests three properties:
1. Random-key collision rate at increasing pair counts (how does the lattice
   handle natural collisions?)
2. Adversarial keys: find key pairs whose addresses collide, plant them in
   the same sequence, measure interference
3. Memory scaling: measure bytes allocated vs unique facts stored
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
from mulattice.kernel import phi_2d

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
CKPT = os.path.join(ROOT, "checkpoints", "capacity8k.pt")


def key_addresses(model, tokens, key_pos):
    """Write addresses of tokens at key_pos."""
    mx = model.blocks[0].mixer
    with torch.no_grad():
        xc = mx.conv(model.blocks[0].n1(model.emb(tokens)))
        kaddr = mx.cfg.addr_scale * torch.tanh(mx.Wa(xc))
    ar = torch.arange(tokens.shape[0], device=tokens.device).unsqueeze(1)
    return kaddr[ar, key_pos]


def test_random_collisions(cfg, model):
    """Natural collision rate at increasing pair counts."""
    results = []
    for n_pairs in [64, 256, 512, 1024, 2048, 4096, 8192]:
        g = torch.Generator().manual_seed(42)
        b = synth_batch(1, n_pairs, 4, cfg, filler=0, generator=g)
        tokens = b["tokens"].to(DEVICE)
        key_pos = b["key_pos"].to(DEVICE)
        addr = key_addresses(model, tokens, key_pos)[0].cpu()

        cells = torch.floor(addr).long()
        codes = cells[:, 0] * 10000 + cells[:, 1]
        unique_cells = len(torch.unique(codes))
        collisions = n_pairs - unique_cells
        density = n_pairs / (2 * cfg.addr_scale + 3) ** 2

        results.append({
            "n_pairs": n_pairs, "unique_cells": unique_cells,
            "collisions": collisions, "collision_rate": collisions / n_pairs,
            "keys_per_cell": density,
        })
        print(f"  {n_pairs:5d} keys: {collisions:4d} collisions "
              f"({collisions/n_pairs:.1%}), {unique_cells:5d} cells, "
              f"density={density:.3f} keys/cell", flush=True)
    return results


def test_adversarial(cfg, model):
    """Find key pairs whose addresses collide, measure interference."""
    g = torch.Generator().manual_seed(123)

    # Generate many keys, compute their addresses, find colliding pairs
    n_candidates = 2000
    key_ids = torch.randperm(cfg.NK, generator=g)[:n_candidates]
    tokens = key_ids.unsqueeze(0).to(DEVICE)
    pos = torch.arange(n_candidates).unsqueeze(0).to(DEVICE)
    addr = key_addresses(model, tokens, pos)[0].cpu()
    cells = torch.floor(addr).long()
    codes = cells[:, 0] * 10000 + cells[:, 1]

    # Find pairs that map to the same cell
    from collections import defaultdict
    cell_map = defaultdict(list)
    for i, c in enumerate(codes.tolist()):
        cell_map[c].append(i)

    colliding_pairs = []
    for c, indices in cell_map.items():
        if len(indices) >= 2:
            for j in range(min(3, len(indices) - 1)):
                colliding_pairs.append((indices[j], indices[j + 1], c))

    print(f"\n  Found {len(colliding_pairs)} colliding key pairs "
          f"from {n_candidates} candidates", flush=True)

    # Measure interference: store both keys, query both
    results = []
    n_test = min(20, len(colliding_pairs))
    for idx in range(n_test):
        i, j, cell = colliding_pairs[idx]
        k1, k2 = int(key_ids[i]), int(key_ids[j])
        v1, v2 = 100, 200  # distinct values

        # Build a small sequence: k1→v1, k2→v2, query k1, query k2
        seq = [k1, v1 + cfg.NK, k2, v2 + cfg.NK, k1, cfg.SEP, k2, cfg.SEP]
        tokens = torch.tensor([seq]).to(DEVICE)
        ans = torch.tensor([[5, 7]]).to(DEVICE)  # SEP positions
        gold = torch.tensor([[v1 + cfg.NK, v2 + cfg.NK]])

        with torch.no_grad():
            pred = logits_at(model, tokens, ans).argmax(-1).cpu()
        ok1 = int(pred[0, 0]) == gold[0, 0]
        ok2 = int(pred[0, 1]) == gold[0, 1]
        results.append({"k1": k1, "k2": k2, "cell": cell,
                        "k1_correct": bool(ok1), "k2_correct": bool(ok2)})

    if results:
        acc1 = sum(r["k1_correct"] for r in results) / len(results)
        acc2 = sum(r["k2_correct"] for r in results) / len(results)
        print(f"  Adversarial collision test ({n_test} pairs): "
              f"key1 recall={acc1:.3f}, key2 recall={acc2:.3f}", flush=True)
    return results


def test_memory_scaling(cfg, model):
    """Measure bytes allocated vs unique facts stored."""
    from mulattice.lattice import LatticeMemory

    results = []
    for n_pairs in [64, 256, 1024, 4096, 8192, 16384]:
        g = torch.Generator().manual_seed(99)
        keys = torch.randperm(cfg.NK, generator=g)[:n_pairs]
        vals = torch.randperm(cfg.NV, generator=g)[:n_pairs]

        # Compute addresses for these keys
        tokens = keys.unsqueeze(0).to(DEVICE)
        pos = torch.arange(n_pairs).unsqueeze(0).to(DEVICE)
        addr = key_addresses(model, tokens, pos)[0]

        # Count unique cells
        cells = torch.floor(addr).long()
        codes = cells[:, 0] * 10000 + cells[:, 1]
        unique_cells = len(torch.unique(codes.cpu()))

        bytes_per_cell = 2 * 32 * 32 * 4  # 2 heads × 32×32 × 4 bytes
        total_bytes = unique_cells * bytes_per_cell

        results.append({
            "n_pairs": n_pairs, "unique_cells": unique_cells,
            "bytes": total_bytes, "mb": total_bytes / 1e6,
            "bytes_per_pair": total_bytes / n_pairs,
        })
        print(f"  {n_pairs:5d} pairs → {unique_cells:5d} cells → "
              f"{total_bytes/1e6:.1f} MB ({total_bytes/n_pairs:.0f} B/pair)",
              flush=True)

        if n_pairs >= 16384:
            # Extrapolate to 1M and 131K pairs
            cells_per_pair = unique_cells / n_pairs
            for target in [131072, 1000000]:
                est_cells = int(target * cells_per_pair)
                est_bytes = est_cells * bytes_per_cell
                print(f"    extrapolated {target:,} pairs → "
                      f"{est_cells:,} cells → {est_bytes/1e9:.1f} GB", flush=True)
    return results


def main():
    cfg = SynthCfg()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg.addr_scale = float(ck["proto_cfg"]["addr_scale"])
    model = build_model(cfg)
    model.load_state_dict(ck["model"])
    model = model.to(DEVICE).eval()
    force_fast_scan(model)

    print("=" * 60)
    print("1. NATURAL COLLISION RATE (random keys)")
    print("=" * 60)
    collision_results = test_random_collisions(cfg, model)

    print("\n" + "=" * 60)
    print("2. ADVERSARIAL COLLISIONS (keys mapped to same cell)")
    print("=" * 60)
    adversarial_results = test_adversarial(cfg, model)

    print("\n" + "=" * 60)
    print("3. MEMORY SCALING")
    print("=" * 60)
    memory_results = test_memory_scaling(cfg, model)

    out = os.path.join(ROOT, "results", "addressing_characterization.json")
    with open(out, "w") as f:
        json.dump({
            "collisions": collision_results,
            "adversarial": adversarial_results,
            "memory": memory_results,
        }, f, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
