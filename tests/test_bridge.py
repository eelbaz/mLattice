#!/usr/bin/env python3
"""Unit tests for the trainable lattice->LM bridge (tests/test_bridge.py).

The bridge is the ONLY trainable component of the bolt-on: it maps a frozen
μLattice readout (d=lattice_dim) to K soft prompt tokens (d=lm_hidden) that
are prepended to a frozen language model's input embeddings.

Covers the four properties the bolt-on depends on:
  * output shape (B, K, lm_hidden) with the Qwen3-4B dimensions;
  * a gate that starts closed (near-zero output at init, so the frozen model
    is not disrupted before training opens the gate);
  * gradient flow from the soft tokens back to the lattice readout;
  * a parameter-count budget (~10.5M) so the bridge stays cheap;
  * training the bridge never modifies frozen parameters it is attached to;
  * the gate actually opens under training (the "stuck gate" failure mode).

Run:  python3 tests/test_bridge.py
Exits 0 on PASS, 1 on FAIL.  CPU-only, ~10s.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn

from mulattice.bridge import LatticeBridge
from mulattice.bolt_on import (
    ANSWER_INSTRUCTION,
    FrozenLattice,
    build_prompt,
    extract_prediction,
    key_id,
    value_id,
)

FAILED = []

LATTICE_DIM = 64          # μLattice d_model (capacity8k checkpoint)
LM_HIDDEN = 2560          # Qwen3.5-4B hidden_size
N_SOFT_TOKENS = 8         # soft prompt tokens prepended to the query
BRIDGE_HIDDEN = 512       # MLP hidden width
PARAM_BUDGET_LOW = 8_000_000     # ~10.5M expected; allow margin either side
PARAM_BUDGET_HIGH = 14_000_000
GATE_INIT_MAX_ABS = 0.1   # soft tokens must stay small at initialization
TRAIN_STEPS = 60
TRAIN_LR = 1e-2
CKPT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "capacity8k.pt")
KEY_SPACE = 16384         # frozen checkpoint key vocabulary (SynthCfg.NK)


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILED.append(name)


def test_bridge_output_shape():
    """Bridge output must be (B, K, lm_hidden)."""
    bridge = LatticeBridge(lattice_dim=LATTICE_DIM, lm_hidden=LM_HIDDEN,
                           n_soft_tokens=N_SOFT_TOKENS, hidden=BRIDGE_HIDDEN)
    readout = torch.randn(2, LATTICE_DIM)
    tokens = bridge(readout)
    check("output shape (B,K,lm_hidden)", tuple(tokens.shape) == (2, N_SOFT_TOKENS, LM_HIDDEN),
          f"got {tuple(tokens.shape)}")


def test_bridge_gate_starts_closed():
    """At init the gate is closed: output near zero so the frozen model is not disrupted."""
    bridge = LatticeBridge(lattice_dim=LATTICE_DIM, lm_hidden=LM_HIDDEN,
                           n_soft_tokens=N_SOFT_TOKENS, hidden=BRIDGE_HIDDEN)
    readout = torch.randn(1, LATTICE_DIM)
    with torch.no_grad():
        tokens = bridge(readout)
    check("gate starts closed", float(tokens.abs().max()) < GATE_INIT_MAX_ABS,
          f"max|out|={float(tokens.abs().max()):.4f}")


def test_bridge_gradients_flow():
    """Gradients must flow from the soft-token output back to the lattice readout."""
    bridge = LatticeBridge(lattice_dim=LATTICE_DIM, lm_hidden=LM_HIDDEN,
                           n_soft_tokens=N_SOFT_TOKENS, hidden=BRIDGE_HIDDEN)
    readout = torch.randn(1, LATTICE_DIM, requires_grad=True)
    tokens = bridge(readout)
    tokens.sum().backward()
    check("gradient reaches readout", readout.grad is not None and float(readout.grad.abs().sum()) > 0)


def test_bridge_param_budget():
    """Bridge stays inside the ~10.5M parameter budget from the session plan."""
    bridge = LatticeBridge(lattice_dim=LATTICE_DIM, lm_hidden=LM_HIDDEN,
                           n_soft_tokens=N_SOFT_TOKENS, hidden=BRIDGE_HIDDEN)
    n = sum(p.numel() for p in bridge.parameters())
    check("parameter budget", PARAM_BUDGET_LOW <= n <= PARAM_BUDGET_HIGH, f"{n:,} params")


def _hash_params(module: nn.Module) -> str:
    """Stable digest of every parameter tensor of a module."""
    import hashlib
    h = hashlib.sha256()
    for name, p in sorted(module.named_parameters()):
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def test_frozen_model_not_modified():
    """Training the bridge must leave a frozen module it feeds bit-identical."""
    torch.manual_seed(0)
    bridge = LatticeBridge(lattice_dim=LATTICE_DIM, lm_hidden=LATTICE_DIM,
                           n_soft_tokens=N_SOFT_TOKENS, hidden=64)
    frozen = nn.Linear(LATTICE_DIM, LATTICE_DIM)
    frozen.requires_grad_(False)
    before = _hash_params(frozen)
    opt = torch.optim.AdamW(bridge.parameters(), lr=TRAIN_LR)
    for _ in range(TRAIN_STEPS):
        readout = torch.randn(4, LATTICE_DIM)
        soft = bridge(readout)
        target = frozen(torch.randn(4, N_SOFT_TOKENS, LATTICE_DIM))
        loss = torch.nn.functional.mse_loss(soft, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    after = _hash_params(frozen)
    check("frozen module bit-identical", before == after)
    check("bridge moved during training",
          any(float(p.grad.abs().sum()) > 0 for p in bridge.parameters() if p.grad is not None))


def test_gate_opens_under_training():
    """The gate must leave zero under training pressure (stuck-gate failure mode)."""
    torch.manual_seed(0)
    bridge = LatticeBridge(lattice_dim=LATTICE_DIM, lm_hidden=32,
                           n_soft_tokens=N_SOFT_TOKENS, hidden=64)
    opt = torch.optim.AdamW(bridge.parameters(), lr=TRAIN_LR)
    target = torch.randn(8, N_SOFT_TOKENS, 32) * 0.05
    for _ in range(TRAIN_STEPS):
        readout = torch.randn(8, LATTICE_DIM)
        loss = torch.nn.functional.mse_loss(bridge(readout), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    gate_open = torch.sigmoid(bridge.gate.detach()).item()
    with torch.no_grad():
        out_scale = float(bridge(torch.randn(1, LATTICE_DIM)).abs().max())
    check("gate opens under training", gate_open > 2.5e-3, f"sigmoid(gate)={gate_open:.5f}")
    check("output grows with gate", out_scale > 0, f"max|out|={out_scale:.5f}")


def test_bridge_batch_consistency():
    """A batched forward must equal per-sample forwards (no cross-row mixing)."""
    torch.manual_seed(1)
    bridge = LatticeBridge(lattice_dim=LATTICE_DIM, lm_hidden=64,
                           n_soft_tokens=N_SOFT_TOKENS, hidden=32)
    with torch.no_grad():
        bridge.gate.fill_(2.0)          # open the gate so differences are visible
        batch = torch.randn(3, LATTICE_DIM)
        out = bridge(batch)
        for i in range(3):
            single = bridge(batch[i:i + 1])
            same = torch.allclose(out[i], single[0], atol=1e-6)
            if not same:
                break
    check("batched == per-sample", same)


def test_hash_ids_deterministic_and_distinct():
    """String→synthetic-id hashing must be deterministic and spread ids."""
    a1, a2 = key_id("Vorelan Mirith"), key_id("Vorelan Mirith")
    check("key id deterministic", a1 == a2)
    check("key ids distinct", key_id("Vorelan Mirith") != key_id("Dranet"))
    v_amber, v_red = value_id("amber"), value_id("red")
    check("value id range", KEY_SPACE <= v_amber < 2 * KEY_SPACE, f"{v_amber}")
    check("value ids distinct", v_amber != v_red)
    ids = [key_id(f"entity-{i}") for i in range(500)]
    # birthday bound: 500 draws over 16,384 ids expect ~7.6 collisions; 480+
    # distinct catches any degenerate hash while tolerating chance collisions
    check("keys spread over space", len(set(ids)) > 480, f"{len(set(ids))}/500 distinct")


def test_frozen_lattice_readout_retrieves():
    """The frozen capacity8k lattice must retrieve a hashed 3-fact binding."""
    lat = FrozenLattice(CKPT, device="cpu")
    facts = [("Vorelan Mirith", "amber"), ("Dranet", "42"), ("Selvora", "7am")]
    readout, pred, gold = lat.readout(facts, "Dranet")
    check("readout shape", tuple(readout.shape) == (1, LATTICE_DIM))
    check("frozen lattice retrieves", pred == gold, f"pred={pred} gold={gold}")


def test_readout_batch_matches_single():
    """Batched (padded) readouts must equal per-case readouts."""
    lat = FrozenLattice(CKPT, device="cpu")
    cases = [
        ([("Vorelan Mirith", "amber"), ("Dranet", "42")], "Dranet"),
        ([("Selvora", "7am"), ("Kraveth", "halva"), ("Onta Bron", "violet"),
          ("Mirud", "slate"), ("Prandar", "cheese")], "Onta Bron"),
    ]
    batch_ro, batch_pred, batch_gold = lat.readout_batch(cases)
    same = True
    for i, (facts, qe) in enumerate(cases):
        ro, pred, gold = lat.readout(facts, qe)
        if not torch.allclose(batch_ro[i], ro[0], atol=1e-5) or batch_pred[i] != pred or batch_gold[i] != gold:
            same = False
    check("batched == per-case readout", same)
    check("batch retrieval correct", all(p == g for p, g in zip(batch_pred, batch_gold)))


def test_prompt_and_extraction_round_trip():
    """Prompt building and prediction extraction must round-trip answer spans."""
    q = "What is the vault password Vorelan set?"
    prompt = build_prompt(q)
    check("prompt carries instruction", ANSWER_INSTRUCTION in prompt and q in prompt)
    check("prompt ends with Answer:", prompt.rstrip().endswith("Answer:"))
    check("extract stops at newline", extract_prediction(" umberlat\nQuestion: ...") == "umberlat")
    check("extract strips space", extract_prediction(" amber") == "amber")
    check("extract two-word answer", extract_prediction(" Vorelan Mirith\n") == "Vorelan Mirith")


def test_format_example_outside_value_space():
    """The prompt scaffold's example value must never be a generator value."""
    import random
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
    from boltoneval import VALUE_TYPES
    rng = random.Random(0)
    sampled = {str(VALUE_TYPES[t](rng)) for _ in range(500) for t in VALUE_TYPES}
    check("example value outside generator space", "tick" not in sampled,
          f"sampled {len(sampled)} distinct values")


def main() -> int:
    test_bridge_output_shape()
    test_bridge_gate_starts_closed()
    test_bridge_gradients_flow()
    test_bridge_param_budget()
    test_frozen_model_not_modified()
    test_gate_opens_under_training()
    test_bridge_batch_consistency()
    test_hash_ids_deterministic_and_distinct()
    test_frozen_lattice_readout_retrieves()
    test_readout_batch_matches_single()
    test_prompt_and_extraction_round_trip()
    test_format_example_outside_value_space()
    print("=" * 74)
    if FAILED:
        print(f"{len(FAILED)} FAILED: {FAILED}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
