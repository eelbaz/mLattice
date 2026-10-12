"""MQAR (multi-query associative recall) data generator, Zoology-style.

A sequence is  key_1 val_1 ... key_N val_N  followed by a query block of Q
keys whose values must be recalled.  Train with <=64 pairs, evaluate with the
same generator at a larger pair count (the published protocol: train <=64,
extrapolate to 256, and probe capacity/forgetting at 2^17).
"""

from __future__ import annotations

import torch


def make_mqar_batch(
    batch: int,
    n_pairs: int,
    n_queries: int,
    n_keys: int,
    n_values: int,
    device="cpu",
    generator: torch.Generator | None = None,
    key_offset: int = 0,
) -> dict:
    """Returns dict with `tokens` (B, L) and `labels` (B, L) (-100 where ignored).

    Keys are sampled WITHOUT replacement per sequence (the standard MQAR protocol):
    every key in a sequence is distinct, so every query has a unique answer and the
    achievable accuracy is 1.0.  Sampling WITH replacement makes later keys collide --
    at 256 pairs over 1024 keys ~30 keys repeat with different values, capping accuracy
    at ~0.885, measuring the ambiguity ceiling instead of the architecture.
    """
    g = generator
    B = batch
    if n_pairs > n_keys:
        raise ValueError(f"n_pairs ({n_pairs}) must be <= n_keys ({n_keys}) to sample distinct keys")
    keys = torch.stack([torch.randperm(n_keys, generator=g)[:n_pairs] for _ in range(B)])
    keys = keys.to(device)
    if n_pairs > n_values:
        raise ValueError(f"n_pairs ({n_pairs}) must be <= n_values ({n_values}) to sample distinct values")
    vals = torch.stack([torch.randperm(n_values, generator=g)[:n_pairs] for _ in range(B)])
    vals = vals.to(device)

    # query positions: distinct keys from the stored set (recall, not lookup-miss)
    qpos = torch.randint(0, n_pairs, (B, n_queries), device=device, generator=g)
    qkeys = torch.gather(keys, 1, qpos)

    kv = torch.stack([keys + key_offset, vals + key_offset + n_keys], dim=-1).reshape(B, 2 * n_pairs)
    qseq = torch.stack(
        [qkeys + key_offset, torch.full_like(qkeys, n_keys + n_values + key_offset + 1)], dim=-1
    ).reshape(B, 2 * n_queries)
    tokens = torch.cat([kv, qseq], dim=1)

    labels = torch.full_like(tokens, -100)
    ans_pos = 2 * n_pairs + torch.arange(1, 2 * n_queries, 2, device=device)
    answers = torch.gather(vals, 1, qpos) + key_offset + n_keys
    labels[:, ans_pos] = answers
    return {"tokens": tokens, "labels": labels, "n_pairs": n_pairs, "answers": answers, "ans_pos": ans_pos}


def vocab_size(n_keys: int, n_values: int) -> int:
    return n_keys + n_values + 2  # keys | values | separator
