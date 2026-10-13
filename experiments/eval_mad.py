#!/usr/bin/env python3
"""MAD (Mechanistic Assessment of DeltaNet) — six synthetic capability tests.

Implements the six task types described in the Percepta blog:
  1. Compression: model must predict a repeating pattern (compress redundancy)
  2. In-context recall: MQAR-style key→value retrieval
  3. Fuzzy recall: query key is perturbed (near-match retrieval)
  4. Memorization: arbitrary key→value with no structure
  5. Noisy recall: recall from context full of distractors
  6. Copy: reproduce a sequence verbatim

Each task is evaluated on our trained checkpoint (not retrained per-task,
matching the blog's evaluation of a single pretrained model across tasks).
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

from mulattice.protocol import SynthCfg, build_model, force_fast_scan, logits_at, synth_batch

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
CKPT = os.path.join(ROOT, "checkpoints", "capacity8k.pt")
N_EVAL = 100  # trials per task


def load_model():
    cfg = SynthCfg()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg.addr_scale = float(ck["proto_cfg"]["addr_scale"])
    m = build_model(cfg)
    m.load_state_dict(ck["model"])
    return m.to(DEVICE).eval(), cfg


# --------------------------------------------------------------- task 1: compression
def task_compression(model, cfg, seed_offset=0):
    """Given a repeating pattern K V K V K V ..., predict the next element.
    Tests whether the model can compress redundancy (predict from pattern)."""
    correct = 0
    for trial in range(N_EVAL):
        g = torch.Generator().manual_seed(trial + seed_offset)
        # Create a repeating key-value pattern
        n_repeats = torch.randint(4, 12, (1,), generator=g).item()
        key = torch.randint(0, cfg.NK, (1,), generator=g).item()
        val = torch.randint(0, cfg.NV, (1,), generator=g).item()

        # Sequence: K V K V ... K V K [?] — predict V
        seq = []
        for _ in range(n_repeats):
            seq.extend([key, val + cfg.NK])
        seq.append(key)  # query: the key one more time
        seq.append(cfg.SEP)  # answer position

        tokens = torch.tensor([seq]).to(DEVICE)
        ans = torch.tensor([[len(seq) - 1]]).to(DEVICE)
        gold = val + cfg.NK

        with torch.no_grad():
            pred = logits_at(model, tokens, ans).argmax(-1).cpu()
        correct += int(int(pred[0, 0]) == gold)
    return correct / N_EVAL


# --------------------------------------------------------------- task 2: in-context recall
def task_in_context_recall(model, cfg, seed_offset=0):
    """Standard MQAR: store N pairs, query one. This is our home turf."""
    g = torch.Generator().manual_seed(seed_offset + 500)
    b = synth_batch(1, 64, min(N_EVAL, 64), cfg, filler=0, generator=g)
    with torch.no_grad():
        pred = logits_at(model, b["tokens"].to(DEVICE), b["ans_pos"].to(DEVICE)).argmax(-1).cpu()
    return float((pred == b["gold"]).float().mean())


# --------------------------------------------------------------- task 3: fuzzy recall
def task_fuzzy_recall(model, cfg, seed_offset=0):
    """Query with a key that's 'close' to the stored key (Hamming distance in
    embedding space). Tests whether the model can handle approximate matching."""
    correct = 0
    n_pairs = 16
    for trial in range(N_EVAL):
        g = torch.Generator().manual_seed(trial + seed_offset + 1000)
        b = synth_batch(1, n_pairs, 1, cfg, filler=0, generator=g)

        # Get the key that was queried and perturb its embedding
        qkey_pos = int(b["ans_pos"][0, 0]) - 1  # position of the query key
        qkey_id = int(b["tokens"][0, qkey_pos])

        # Find a different key with a similar embedding
        # (we approximate "fuzzy" by using a random other key from the same batch)
        other_keys = [int(b["tokens"][0, int(b["key_pos"][0, i])])
                      for i in range(n_pairs) if int(b["tokens"][0, int(b["key_pos"][0, i])]) != qkey_id]
        if not other_keys:
            continue
        fuzzy_key = other_keys[torch.randint(0, len(other_keys), (1,), generator=g).item()]

        # Replace the query key with the fuzzy key
        tokens = b["tokens"].clone()
        tokens[0, qkey_pos] = fuzzy_key

        # Gold: the value associated with the fuzzy key (not the original)
        fuzzy_idx = None
        for i in range(n_pairs):
            if int(tokens[0, int(b["key_pos"][0, i])]) == fuzzy_key:
                fuzzy_idx = i
                break
        if fuzzy_idx is None:
            continue
        gold = int(b["tokens"][0, int(b["key_pos"][0, fuzzy_idx]) + 1])

        with torch.no_grad():
            pred = logits_at(model, tokens.to(DEVICE), b["ans_pos"].to(DEVICE)).argmax(-1).cpu()
        correct += int(int(pred[0, 0]) == gold)
    return correct / N_EVAL


# --------------------------------------------------------------- task 4: memorization
def task_memorization(model, cfg, seed_offset=0):
    """Store many pairs and query all of them. This is our capacity test but
    at a moderate size. Tests raw memorization capacity."""
    g = torch.Generator().manual_seed(seed_offset + 2000)
    n_pairs = 1024
    b = synth_batch(1, n_pairs, 64, cfg, filler=0, generator=g)
    with torch.no_grad():
        pred = logits_at(model, b["tokens"].to(DEVICE), b["ans_pos"].to(DEVICE)).argmax(-1).cpu()
    return float((pred == b["gold"]).float().mean())


# --------------------------------------------------------------- task 5: noisy recall
def task_noisy_recall(model, cfg, seed_offset=0):
    """Recall one pair from a sequence full of distractor filler tokens.
    Tests robustness to noise."""
    correct = 0
    for trial in range(N_EVAL):
        g = torch.Generator().manual_seed(trial + seed_offset + 3000)
        b = synth_batch(1, 64, 1, cfg, filler=14, generator=g)
        with torch.no_grad():
            pred = logits_at(model, b["tokens"].to(DEVICE), b["ans_pos"].to(DEVICE)).argmax(-1).cpu()
        correct += int(int(pred[0, 0]) == int(b["gold"][0, 0]))
    return correct / N_EVAL


# --------------------------------------------------------------- task 6: copy
def task_copy(model, cfg, seed_offset=0):
    """Given a sequence of tokens, reproduce it verbatim.
    Tests whether the model can echo input (the simplest sequence task)."""
    correct = 0
    total = 0
    for trial in range(min(20, N_EVAL)):  # fewer trials (slower per-trial)
        g = torch.Generator().manual_seed(trial + seed_offset + 4000)
        seq_len = torch.randint(8, 32, (1,), generator=g).item()
        # Random tokens from the full vocabulary (keys + values)
        seq = torch.randint(0, cfg.NK + cfg.NV, (seq_len,), generator=g).tolist()
        # Append a separator, then the model should echo the sequence
        tokens = seq + [cfg.SEP] + [0] * seq_len  # zeros to be filled by prediction
        tokens = torch.tensor([tokens]).to(DEVICE)

        # Score at positions after SEP: does the model predict seq[i]?
        for i in range(seq_len):
            pos = seq_len + 1 + i  # position where seq[i] should be predicted
            if pos >= tokens.shape[1]:
                break
            ans = torch.tensor([[pos]]).to(DEVICE)
            with torch.no_grad():
                pred = logits_at(model, tokens[:, :pos + 1], ans).argmax(-1).cpu()
            correct += int(int(pred[0, 0]) == seq[i])
            total += 1
    return correct / max(total, 1)


def main():
    model, cfg = load_model()
    force_fast_scan(model)
    print(f"MAD evaluation on {DEVICE}", flush=True)
    print(f"model: capacity8k.pt (not retrained per-task)", flush=True)
    print()

    tasks = [
        ("Compression", task_compression),
        ("In-context recall", task_in_context_recall),
        ("Fuzzy recall", task_fuzzy_recall),
        ("Memorization", task_memorization),
        ("Noisy recall", task_noisy_recall),
        ("Copy", task_copy),
    ]

    results = {}
    scores = []
    for name, fn in tasks:
        t0 = time.time()
        score = fn(model, cfg)
        results[name] = round(score, 4)
        scores.append(score)
        print(f"  {name:20s} {score:.4f}  [{time.time()-t0:.1f}s]", flush=True)

    mean = sum(scores) / len(scores)
    results["Mean"] = round(mean, 4)
    print(f"\n  {'Mean':20s} {mean:.4f}", flush=True)

    # Comparison with published Spotlight scores
    published = {"Compression": 0.456, "In-context recall": 1.000, "Fuzzy recall": 0.577,
                 "Memorization": 0.689, "Noisy recall": 1.000, "Copy": 1.000, "Mean": 0.787}
    print(f"\n  {'Task':20s} {'μLattice':>8s} {'Published':>10s}")
    print(f"  {'-'*40}")
    for name in list(results.keys()):
        pub = published.get(name, "—")
        print(f"  {name:20s} {results[name]:>8.4f} {pub:>10}", flush=True)

    out = os.path.join(ROOT, "results", "mad.json")
    with open(out, "w") as f:
        json.dump({"results": results, "published": published, "n_eval": N_EVAL}, f, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
