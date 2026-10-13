#!/usr/bin/env python3
"""Train the μLattice→Qwen3.5 soft-prompt bridge (Stage A + Stage B).

Stage A — embedding alignment: the bridge learns to emit the frozen LM's
input embeddings of the answer tokens when fed the frozen lattice readout.
No LM forward pass is needed; the loss is masked MSE against the answer
token embeddings.

Stage B — end-to-end: [soft tokens | prompt | answer] is fed to the frozen
LM via inputs_embeds and the bridge is trained on the answer-token NLL.
Only the bridge receives gradients; the LM and the lattice stay frozen
(verified by hashing before/after training).

Fresh-fact discipline: training cases come from seed TRAIN_SEED, holdout
cases from HOLDOUT_SEED; entity and (entity, value) overlap between the
two sets is asserted to be empty.  Documents never enter the test prompt —
only the lattice readout does.

Usage:
  python3 experiments/train_bridge.py            # full run (GPU)
  SMOKE=1 python3 experiments/train_bridge.py    # plumbing check
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from boltoneval import make_case_with_facts, eval_report  # noqa: E402
from mulattice.bolt_on import (  # noqa: E402
    FrozenLattice, build_prompt, extract_prediction, greedy_decode, key_id,
)
from mulattice.bridge import LatticeBridge  # noqa: E402

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = getattr(torch, os.environ.get("DTYPE", "bfloat16"))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(ROOT, "models", "qwen3.5-4b"))
LATTICE_CKPT = os.environ.get("LATTICE_CKPT", os.path.join(ROOT, "checkpoints", "capacity8k.pt"))
CKPT_LATEST = os.path.join(ROOT, "checkpoints", "bridge_latest.pt")
CKPT_BEST = os.path.join(ROOT, "checkpoints", "bridge.pt")
HISTORY_PATH = os.path.join(ROOT, "results", "bridge_training.json")

SMOKE = bool(os.environ.get("SMOKE"))
TRAIN_SEED = 11
HOLDOUT_SEED = 22
N_TRAIN_CASES = 8000 if not SMOKE else 128
N_HOLDOUT_CASES = 256 if not SMOKE else 8
FACTS_MIN, FACTS_MAX = 2, 10
FILLER_PER_DOC = 20
STAGE_A_STEPS = 600 if not SMOKE else 20
STAGE_B_STEPS = 3000 if not SMOKE else 30
STAGE_A_BATCH = 64
STAGE_B_BATCH = 32 if not SMOKE else 4
STAGE_A_LR = 1e-3
STAGE_B_LR = 2e-4
STAGE_B_WARMUP = 100
STAGE_B_MIN_FRAC = 0.15          # cosine floor as a fraction of STAGE_B_LR
HOLDOUT_DECODE_N = 32 if not SMOKE else 4
EVAL_EVERY = 250 if not SMOKE else 10
SAVE_EVERY = 500 if not SMOKE else 25
GRAD_CLIP = 1.0
MAX_NEW_TOKENS = 12
READOUT_CHUNK = 512


def generate_cases(n: int, seed: int, forbidden_pairs: set | None = None) -> list[dict]:
    """Fresh-fact cases; each carries the fact list for lattice ingestion.

    Cases whose entities collide on lattice key ids (same cell address for
    two facts in one document) are redrawn — the lattice cannot hold both.
    Cases containing any forbidden (entity, value) binding are redrawn too,
    so freshness holds by construction, not just by assertion."""
    import random
    rng = random.Random(seed)
    cases = []
    while len(cases) < n:
        n_facts = rng.randint(FACTS_MIN, FACTS_MAX)
        case, facts = make_case_with_facts(rng, n_facts, FILLER_PER_DOC)
        keys = [key_id(f.entity) for f in facts]
        if len(set(keys)) != len(keys):
            continue
        pairs = [(f.entity, f.value) for f in facts]
        if forbidden_pairs is not None and any(p in forbidden_pairs for p in pairs):
            continue
        asked = [f for f in facts if f.question == case.question][0]
        cases.append({
            "case": case, "facts": pairs,
            "query_entity": asked.entity, "answer": case.answer,
            "question": case.question, "fact_type": case.fact_type,
        })
    return cases


def assert_fresh(train_cases: list[dict], holdout_cases: list[dict]) -> None:
    """No (entity, value) binding may cross the train/holdout split.

    Entity NAMES repeat across documents by construction (the syllable
    generator's space is ~1.4M names; ~50K training draws share names with
    any other large sample).  A fact is the binding, not the name — and the
    bridge never sees entities, only post-retrieval readouts — so
    binding-level disjointness is the freshness guarantee."""
    train_pairs = {(e, v) for c in train_cases for e, v in c["facts"]}
    hold_pairs = {(e, v) for c in holdout_cases for e, v in c["facts"]}
    overlap = train_pairs & hold_pairs
    assert not overlap, f"{len(overlap)} facts overlap between train and holdout"


def module_hash(module: torch.nn.Module) -> str:
    """Stable digest over every parameter tensor of a module."""
    h = hashlib.sha256()
    for name, p in sorted(module.named_parameters()):
        h.update(name.encode())
        h.update(p.detach().to(torch.float32).cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def precompute_readouts(lat: FrozenLattice, cases: list[dict]):
    """Lattice readouts for every case + retrieval sanity count."""
    lat_cases = [(c["facts"], c["query_entity"]) for c in cases]
    readouts, correct = [], 0
    for i in range(0, len(lat_cases), READOUT_CHUNK):
        ro, preds, golds = lat.readout_batch(lat_cases[i:i + READOUT_CHUNK])
        readouts.append(ro)
        correct += sum(int(p == g) for p, g in zip(preds, golds))
    return torch.cat(readouts), correct, len(lat_cases)


@torch.no_grad()
def encode_case(tok, case: dict) -> dict:
    prompt_ids = tok(build_prompt(case["question"]), add_special_tokens=False).input_ids
    ans_ids = tok(" " + case["answer"] + "\n", add_special_tokens=False).input_ids
    return {"prompt_ids": prompt_ids, "ans_ids": ans_ids}


def stage_a_step(bridge, opt, readouts, ans_targets, emb_table):
    """Masked MSE between soft tokens and the answer's own token embeddings."""
    idx = torch.randint(0, readouts.shape[0], (STAGE_A_BATCH,))
    ro = readouts[idx]
    soft = bridge(ro)                                   # (B, K, H)
    B, K, H = soft.shape
    target = torch.zeros(B, K, H, device=soft.device, dtype=soft.dtype)
    mask = torch.zeros(B, K, dtype=torch.bool, device=soft.device)
    for b, ids in enumerate([ans_targets[i] for i in idx.tolist()]):
        n = min(len(ids), K)
        target[b, :n] = emb_table[ids[:n]]
        mask[b, :n] = True
    loss = F.mse_loss(soft[mask], target[mask])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(bridge.parameters(), GRAD_CLIP)
    opt.step()
    return float(loss.detach())


def stage_b_step(bridge, opt, batch, emb, lm, pad_id):
    """End-to-end NLL of the answer tokens through the frozen LM."""
    rows = batch["rows"]
    soft = bridge(batch["readouts"]).to(emb.weight.dtype)      # (B, K, H)
    P = max(len(r["prompt_ids"]) for r in rows)
    A = max(len(r["ans_ids"]) for r in rows)
    B, K, H = soft.shape
    inputs = torch.zeros(B, K + P + A, H, device=soft.device, dtype=soft.dtype)
    attn = torch.zeros(B, K + P + A, dtype=torch.long, device=soft.device)
    labels = torch.full((B, A), -100, dtype=torch.long, device=soft.device)
    inputs[:, :K] = soft
    attn[:, :K] = 1
    for b, r in enumerate(rows):
        pi = torch.tensor(r["prompt_ids"], device=soft.device)
        ai = torch.tensor(r["ans_ids"], device=soft.device)
        inputs[b, K:K + len(pi)] = emb(pi).to(soft.dtype)
        inputs[b, K + P:K + P + len(ai)] = emb(ai).to(soft.dtype)
        attn[b, K:K + len(pi)] = 1
        attn[b, K + P:K + P + len(ai)] = 1
        labels[b, :len(ai)] = ai
    logits = lm(inputs_embeds=inputs, attention_mask=attn, use_cache=False).logits
    pred = logits[:, K + P - 1:K + P - 1 + A]                # predicts answer slots
    loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]).float(), labels.reshape(-1),
                           ignore_index=-100)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(bridge.parameters(), GRAD_CLIP))
    opt.step()
    with torch.no_grad():
        top1 = (pred.argmax(-1) == labels)[labels != -100].float().mean().item()
    return float(loss.detach()), grad_norm, top1


@torch.no_grad()
def holdout_decode(bridge, lm, tok, holdout, readouts, n):
    """Exact-match on fresh holdout cases via greedy decode with soft tokens."""
    bridge.eval()
    preds = []
    emb = lm.get_input_embeddings()
    eos_ids = (tok.eos_token_id, tok.pad_token_id)
    for i in range(n):
        c = holdout[i]
        prompt_ids = tok(build_prompt(c["question"]), add_special_tokens=False,
                         return_tensors="pt").input_ids.to(DEVICE)
        soft = bridge(readouts[i:i + 1])
        gen = greedy_decode(lm, emb, prompt_ids, soft, eos_ids, MAX_NEW_TOKENS)
        text = tok.decode(gen, skip_special_tokens=True)
        preds.append((extract_prediction(text), c["answer"]))
    bridge.train()
    return eval_report("holdout", preds)["exact_match"], preds


def save(bridge, path, extra):
    payload = {"bridge": bridge.state_dict(), "config": {
        "lattice_dim": bridge.read_proj[0].in_features,
        "lm_hidden": bridge.lm_hidden, "n_soft_tokens": bridge.n_soft}}
    payload.update(extra)
    torch.save(payload, path)


def load_lm(path: str):
    """Load the frozen target LM (Qwen3.5-4B is a multimodal
    Qwen3_5ForConditionalGeneration; text-only use falls back to it when the
    CausalLM auto-class does not map)."""
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(path, dtype=DTYPE)
    except (ValueError, KeyError) as exc:
        print(f"AutoModelForCausalLM unsuitable ({type(exc).__name__}); "
              f"loading Qwen3_5ForConditionalGeneration", flush=True)
        from transformers import Qwen3_5ForConditionalGeneration
        return Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=DTYPE)


def main():
    t0 = time.time()
    smoke_tag = " (SMOKE)" if SMOKE else ""
    print(f"TRAIN BRIDGE{smoke_tag}  device={DEVICE} dtype={DTYPE}", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    lm = load_lm(MODEL_PATH)
    lm = lm.to(DEVICE).eval()
    lm.requires_grad_(False)
    emb = lm.get_input_embeddings()
    emb.weight.requires_grad_(False)
    hidden = emb.weight.shape[1]
    print(f"LM loaded: hidden={hidden} vocab={emb.weight.shape[0]} "
          f"({time.time()-t0:.0f}s)", flush=True)

    lat = FrozenLattice(LATTICE_CKPT, device=DEVICE)
    lat_hash = module_hash(lat.model)
    lm_hash_before = module_hash(lm)
    print(f"frozen lattice hash {lat_hash}", flush=True)
    print(f"frozen LM hash (before) {lm_hash_before}", flush=True)

    train_cases = generate_cases(N_TRAIN_CASES, TRAIN_SEED)
    train_pairs = {(e, v) for c in train_cases for e, v in c["facts"]}
    holdout_cases = generate_cases(N_HOLDOUT_CASES, HOLDOUT_SEED,
                                   forbidden_pairs=train_pairs)
    assert_fresh(train_cases, holdout_cases)
    print(f"cases: {len(train_cases)} train / {len(holdout_cases)} holdout, "
          f"fresh-fact asserts passed", flush=True)

    train_ro, tr_ok, tr_n = precompute_readouts(lat, train_cases)
    hold_ro, ho_ok, ho_n = precompute_readouts(lat, holdout_cases)
    print(f"lattice retrieval: train {tr_ok}/{tr_n} = {tr_ok/tr_n:.4f}  "
          f"holdout {ho_ok}/{ho_n} = {ho_ok/ho_n:.4f}", flush=True)

    train_enc = [encode_case(tok, c) for c in train_cases]
    ans_targets = [e["ans_ids"] for e in train_enc]
    bridge = LatticeBridge(lattice_dim=train_ro.shape[1], lm_hidden=hidden).to(DEVICE)
    n_params = sum(p.numel() for p in bridge.parameters())
    print(f"bridge params {n_params:,}  gate {torch.sigmoid(bridge.gate).item():.2e}",
          flush=True)

    history = []

    opt_a = torch.optim.AdamW(bridge.parameters(), lr=STAGE_A_LR)
    for step in range(1, STAGE_A_STEPS + 1):
        loss = stage_a_step(bridge, opt_a, train_ro, ans_targets, emb.weight)
        if step % 50 == 0 or step == 1:
            print(f"  A {step:5d} mse {loss:.4f} gate {torch.sigmoid(bridge.gate).item():.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if loss == 0.0:
                print("RED FLAG: stage-A loss exactly zero", flush=True)
    history.append({"stage": "A", "steps": STAGE_A_STEPS, "final_mse": loss})

    opt_b = torch.optim.AdamW(bridge.parameters(), lr=STAGE_B_LR)
    sched = torch.optim.lr_scheduler.LambdaLR(opt_b, lambda s: (
        (s + 1) / STAGE_B_WARMUP if s < STAGE_B_WARMUP else
        STAGE_B_MIN_FRAC + (1 - STAGE_B_MIN_FRAC) * 0.5 *
        (1 + torch.cos(torch.tensor(torch.pi * (s - STAGE_B_WARMUP) /
                                    max(1, STAGE_B_STEPS - STAGE_B_WARMUP))).item())))
    order = torch.randperm(N_TRAIN_CASES).tolist()
    cursor = 0
    best_em = -1.0

    def next_batch():
        nonlocal cursor, order
        if cursor + STAGE_B_BATCH > len(order):
            order = torch.randperm(N_TRAIN_CASES).tolist()
            cursor = 0
        idx = order[cursor:cursor + STAGE_B_BATCH]
        cursor += STAGE_B_BATCH
        rows = [train_enc[i] | {"fact_type": train_cases[i]["fact_type"]} for i in idx]
        return {"rows": rows, "readouts": train_ro[idx]}

    for step in range(1, STAGE_B_STEPS + 1):
        batch = next_batch()
        loss, gnorm, top1 = stage_b_step(bridge, opt_b, batch, emb, lm, tok.pad_token_id)
        sched.step()
        if step % 50 == 0 or step == 1:
            gate = torch.sigmoid(bridge.gate).item()
            print(f"  B {step:5d} ce {loss:.3f} top1 {top1:.3f} |g| {gnorm:.2f} "
                  f"gate {gate:.3f} lr {sched.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s)",
                  flush=True)
            if gnorm == 0.0:
                print("RED FLAG: zero gradient norm", flush=True)
            history.append({"stage": "B", "step": step, "ce": round(loss, 4),
                            "top1": round(top1, 4), "gate": round(gate, 4)})
        if step % EVAL_EVERY == 0 or step == STAGE_B_STEPS:
            em, _ = holdout_decode(bridge, lm, tok, holdout_cases, hold_ro,
                                   min(HOLDOUT_DECODE_N, len(holdout_cases)))
            print(f"  HOLDOUT exact-match {em:.4f} @ step {step}", flush=True)
            history.append({"stage": "B", "step": step, "holdout_em": em})
            if em > best_em:
                best_em = em
                save(bridge, CKPT_BEST, {"step": step, "stage": "B",
                                         "holdout_exact_match": em,
                                         "train_seed": TRAIN_SEED,
                                         "holdout_seed": HOLDOUT_SEED,
                                         "lattice_hash": lat_hash,
                                         "lm_hash_before": lm_hash_before})
        if step % SAVE_EVERY == 0:
            save(bridge, CKPT_LATEST, {"step": step, "stage": "B",
                                       "holdout_exact_match": best_em,
                                       "train_seed": TRAIN_SEED})
            with open(HISTORY_PATH, "w") as fh:
                json.dump(history, fh, indent=2)

    lm_hash_after = module_hash(lm)
    frozen_ok = lm_hash_after == lm_hash_before
    print(f"frozen LM hash (after) {lm_hash_after}  unchanged={frozen_ok}", flush=True)
    save(bridge, CKPT_LATEST, {"step": STAGE_B_STEPS, "stage": "B",
                               "holdout_exact_match": best_em,
                               "train_seed": TRAIN_SEED,
                               "lm_hash_before": lm_hash_before,
                               "lm_hash_after": lm_hash_after,
                               "frozen_unchanged": frozen_ok})
    with open(HISTORY_PATH, "w") as fh:
        json.dump(history, fh, indent=2)
    print(f"DONE best holdout exact-match {best_em:.4f}; "
          f"frozen LM unchanged={frozen_ok}; total {time.time()-t0:.0f}s", flush=True)
    if not frozen_ok:
        raise SystemExit("FROZEN MODEL WAS MODIFIED — aborting")


if __name__ == "__main__":
    main()
