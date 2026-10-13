#!/usr/bin/env python3
"""Bolt-on evaluation: floor / ceiling / test / control on fresh facts.

  floor    frozen Qwen3.5 alone — the question with no memory and no document
  ceiling  RAG — the document pasted into the context (the upper anchor)
  test     frozen Qwen3.5 + μLattice soft tokens via the trained bridge;
           the document is NOT in the prompt — it exists only as lattice state
  control  soft tokens from ANOTHER case's lattice readout (shuffled): if the
           bridge steers by content, mismatched memory must not help

Fresh-fact discipline: cases come from --seed (default 999), disjoint from
the training seed (11) and the trainer's holdout seed (22); entity and
(entity, value) overlap with the training set is asserted empty.  Scoring is
exact-match after normalisation (experiments/boltoneval.py eval_report).

Run:  python3 experiments/eval_bolt_on.py [--n 200] [--seed 999]
Writes results/bolt_on_eval.json and results/bolt_on_eval_cases.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from boltoneval import eval_report, make_case_with_facts, norm_answer  # noqa: E402
from mulattice.bolt_on import (  # noqa: E402
    FrozenLattice, build_prompt, build_rag_prompt, extract_prediction, greedy_decode,
    key_id,
)
from mulattice.bridge import LatticeBridge  # noqa: E402

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = getattr(torch, os.environ.get("DTYPE", "bfloat16"))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(ROOT, "models", "qwen3.5-4b"))
LATTICE_CKPT = os.environ.get("LATTICE_CKPT", os.path.join(ROOT, "checkpoints", "capacity8k.pt"))
BRIDGE_CKPT = os.environ.get("BRIDGE_CKPT", os.path.join(ROOT, "checkpoints", "bridge.pt"))
TRAIN_SEED = 11                  # must match experiments/train_bridge.py
FACTS_MIN, FACTS_MAX = 2, 10     # must match the trainer
FILLER_PER_DOC = 20
MAX_NEW_TOKENS = 12
OUT_JSON = os.path.join(ROOT, "results", "bolt_on_eval.json")
OUT_CASES = os.path.join(ROOT, "results", "bolt_on_eval_cases.jsonl")
CONDITIONS = ("floor", "ceiling", "test", "control")


def generate_cases(n: int, seed: int, forbidden_pairs: set | None = None) -> list[dict]:
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
        cases.append({"case": case, "facts": pairs,
                      "query_entity": asked.entity})
    return cases


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-tailed p-value that two proportions differ (pooled z-test)."""
    if n1 == 0 or n2 == 0:
        return 1.0
    p1, p2 = k1 / n1, k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0 if p1 != p2 else 1.0
    z = (p1 - p2) / se
    return math.erfc(abs(z) / math.sqrt(2))


@torch.no_grad()
def run_condition(name: str, cases: list[dict], readouts: torch.Tensor | None,
                  bridge, lm, tok, emb, rag: bool = False) -> list[dict]:
    eos_ids = (tok.eos_token_id, tok.pad_token_id)
    records = []
    for i, c in enumerate(cases):
        q = c["case"].question
        prompt = build_rag_prompt(c["case"].doc, q) if rag else build_prompt(q)
        prompt_ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(DEVICE)
        soft = None
        if name == "test":
            soft = bridge(readouts[i:i + 1])
        elif name == "control":
            j = (i + 1) % len(cases)          # a different case's readout
            soft = bridge(readouts[j:j + 1])
        gen = greedy_decode(lm, emb, prompt_ids, soft, eos_ids, MAX_NEW_TOKENS)
        text = tok.decode(gen, skip_special_tokens=True)
        pred = extract_prediction(text)
        gold = c["case"].answer
        records.append({"condition": name, "fact_type": c["case"].fact_type,
                        "gold": gold, "pred": pred,
                        "correct": int(norm_answer(pred) == norm_answer(gold))})
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=999)
    args = ap.parse_args()
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=DTYPE).to(DEVICE).eval()
    lm.requires_grad_(False)
    emb = lm.get_input_embeddings()

    lat = FrozenLattice(LATTICE_CKPT, device=DEVICE)
    ck = torch.load(BRIDGE_CKPT, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    bridge = LatticeBridge(lattice_dim=cfg["lattice_dim"], lm_hidden=cfg["lm_hidden"],
                           n_soft_tokens=cfg["n_soft_tokens"]).to(DEVICE)
    bridge.load_state_dict(ck["bridge"])
    bridge.eval()
    bridge.requires_grad_(False)
    gate = torch.sigmoid(bridge.gate).item()
    print(f"eval: n={args.n} seed={args.seed} device={DEVICE} "
          f"bridge step={ck.get('step')} holdout_em={ck.get('holdout_exact_match')} "
          f"gate={gate:.3f}", flush=True)

    train_cases = generate_cases(8000, TRAIN_SEED)
    tr_pairs = {(e, v) for c in train_cases for e, v in c["facts"]}
    tr_ents = {e for c in train_cases for e, _ in c["facts"]}
    cases = generate_cases(args.n, args.seed, forbidden_pairs=tr_pairs)
    ev_pairs = {(e, v) for c in cases for e, v in c["facts"]}
    ev_ents = {e for c in cases for e, _ in c["facts"]}
    assert not (tr_pairs & ev_pairs), "eval facts appear in training"
    overlap = {"train_facts": len(tr_pairs), "eval_facts": len(ev_pairs),
               "fact_overlap": 0, "shared_entity_names": len(tr_ents & ev_ents)}
    print(f"fresh-fact asserts passed: {overlap}", flush=True)
    lat_cases = [(c["facts"], c["query_entity"]) for c in cases]
    readouts, correct, total = [], 0, 0
    for i in range(0, len(lat_cases), 512):
        ro, preds, golds = lat.readout_batch(lat_cases[i:i + 512])
        readouts.append(ro)
        correct += sum(int(p == g) for p, g in zip(preds, golds))
        total += len(preds)
    readouts = torch.cat(readouts)
    print(f"lattice retrieval on eval facts: {correct}/{total} = {correct/total:.4f}",
          flush=True)

    all_records = []
    for cond in CONDITIONS:
        recs = run_condition(cond, cases, readouts, bridge, lm, tok, emb,
                             rag=(cond == "ceiling"))
        all_records.extend(recs)
        rep = eval_report(cond, [(r["pred"], r["gold"]) for r in recs])
        print(f"  {cond:8s} exact-match {rep['exact_match']:.4f}", flush=True)

    def acc(cond: str, ftype: str | None = None) -> tuple[int, int]:
        rows = [r for r in all_records if r["condition"] == cond
                and (ftype is None or r["fact_type"] == ftype)]
        return sum(r["correct"] for r in rows), len(rows)

    summary = {}
    for cond in CONDITIONS:
        k, n = acc(cond)
        per_type = {}
        for ftype in sorted({r["fact_type"] for r in all_records}):
            tk, tn = acc(cond, ftype)
            per_type[ftype] = {"correct": tk, "n": tn,
                               "exact_match": round(tk / tn, 4) if tn else None}
        summary[cond] = {"correct": k, "n": n,
                         "exact_match": round(k / n, 4) if n else None,
                         "per_type": per_type}
    tk, tn = acc("test")
    fk, fn = acc("floor")
    ck2, cn = acc("control")
    stats = {
        "test_vs_floor_z": round(two_proportion_z(tk, tn, fk, fn), 8),
        "test_vs_control_z": round(two_proportion_z(tk, tn, ck2, cn), 8),
        "test_minus_floor": round(tk / tn - fk / fn, 4),
    }
    result = {"n": args.n, "seed": args.seed, "model": MODEL_PATH,
              "bridge_checkpoint": BRIDGE_CKPT, "bridge_step": ck.get("step"),
              "gate": round(gate, 4),
              "lattice_retrieval": round(correct / total, 4),
              "fresh_fact_asserts": overlap, "summary": summary, "stats": stats,
              "wallclock_s": round(time.time() - t0, 1)}
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    with open(OUT_CASES, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"summary": {c: summary[c]["exact_match"] for c in CONDITIONS},
                      "stats": stats}, indent=2), flush=True)
    print(f"results -> {OUT_JSON}  ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
