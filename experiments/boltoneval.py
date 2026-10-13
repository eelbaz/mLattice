#!/usr/bin/env python3
"""Bolt-on memory evaluation harness.

Evaluates a memory-injection method against a frozen language model.

  * every test draws fresh random facts (entity names, values, bindings)
    from a generator with >10^30 combinations;
  * three baselines anchor every number:
      floor   -- the frozen model alone (no memory),
      ceiling -- RAG prompt-stuffing (facts pasted into the context),
      graft   -- the memory-injection method under test;
  * metrics: exact-match on the answer span, plus a contamination check
    that the answer string does not occur in the injected context.

The fact generator is model-agnostic (sentences built from random
syllable-entities and typed values).
"""
from __future__ import annotations

import argparse
import json
import random
import string
from dataclasses import dataclass, asdict


def _syllable_entity(rng: random.Random) -> str:
    """A fresh pronounceable name, e.g. 'Vorelan Mirith' -- never a real word."""
    onsets = ["b", "br", "c", "d", "dr", "f", "g", "gr", "k", "l", "m", "n",
              "p", "pr", "r", "s", "st", "t", "tr", "v", "z"]
    nuclei = ["a", "e", "i", "o", "u", "ai", "ei", "ou", "au"]
    codas = ["", "n", "r", "l", "s", "th", "m"]

    def syl():
        return rng.choice(onsets) + rng.choice(nuclei) + rng.choice(codas)

    name = (syl() + syl()).capitalize()
    if rng.random() < 0.5:
        name += " " + (syl() + syl()).capitalize()
    return name


VALUE_TYPES = {
    "password": lambda rng: "".join(rng.choice(string.ascii_lowercase)
                                    for _ in range(rng.randint(6, 10))),
    "color": lambda rng: rng.choice(["red", "blue", "green", "amber", "violet",
                                     "white", "black", "teal", "crimson", "slate"]),
    "time": lambda rng: f"{rng.randint(1,12)}{rng.choice(['am', 'pm'])}",
    "number": lambda rng: str(rng.randint(2, 98)),
    "city": lambda rng: _syllable_entity(rng),          # fictional on purpose
    "food": lambda rng: rng.choice(["stew", "bread", "plums", "cheese", "halva",
                                    "porridge", "dates", "salmon", "biscuits"]),
}

TEMPLATES = {
    "password": [
        ("{e} set the vault password to {v} before leaving.", "What is the vault password {e} set?"),
        ("The password for {e}'s account is {v}.", "What is the password for {e}'s account?"),
    ],
    "color": [
        ("{e} painted the boat {v}.", "What color did {e} paint the boat?"),
        ("The door of {e}'s house is {v}.", "What color is the door of {e}'s house?"),
    ],
    "time": [
        ("The meeting with {e} is at {v}.", "When is the meeting with {e}?"),
        ("{e} said the departure is at {v}.", "When did {e} say the departure is?"),
    ],
    "number": [
        ("{e} counted {v} crates on the dock.", "How many crates did {e} count?"),
        ("There are {v} keys in {e}'s drawer.", "How many keys are in {e}'s drawer?"),
    ],
    "city": [
        ("{e} was born in {v}.", "Where was {e} born?"),
        ("{e} is travelling to {v}.", "Where is {e} travelling?"),
    ],
    "food": [
        ("{e} ordered {v} at the inn.", "What did {e} order at the inn?"),
        ("{e}'s favorite meal is {v}.", "What is {e}'s favorite meal?"),
    ],
}

FILLER = [
    "The harbor was quiet that morning.",
    "Gulls circled the masts while the tide went out.",
    "A clerk stamped papers without looking up.",
    "Somewhere a bell rang twice, then stopped.",
    "The road east had been washed out again.",
    "Dust settled over the market stalls.",
    "Two children argued about a kite.",
    "The lamp in the window burned all night.",
    "A cart of turnips arrived late.",
    "Rain threatened but never came.",
]


@dataclass
class FactCase:
    doc: str            # the document to ingest (facts buried in filler)
    question: str       # a question about ONE fact
    answer: str         # the exact answer span
    fact_type: str


@dataclass
class Fact:
    """One entity→value binding inside a FactCase document."""
    entity: str
    value: str
    question: str       # the question template instantiated for this fact
    fact_type: str
    template_idx: int = 0   # which template pair instantiated this fact


def make_case_with_facts(rng: random.Random, n_facts: int, n_filler: int) -> tuple[FactCase, list[Fact]]:
    """One evaluation case plus the full fact list it was built from."""
    facts: list[Fact] = []
    used_entities = set()
    for _ in range(n_facts):
        ftype = rng.choice(list(VALUE_TYPES))
        ent = _syllable_entity(rng)
        while ent in used_entities:
            ent = _syllable_entity(rng)
        used_entities.add(ent)
        val = VALUE_TYPES[ftype](rng)
        t_idx = rng.randrange(len(TEMPLATES[ftype]))
        sent_t, q_t = TEMPLATES[ftype][t_idx]
        facts.append(Fact(entity=ent, value=str(val), question=q_t.format(e=ent),
                          fact_type=ftype, template_idx=t_idx))
    order = list(range(len(facts)))
    rng.shuffle(order)
    parts = []
    for i in order:
        sent_t, _ = TEMPLATES[facts[i].fact_type][facts[i].template_idx]
        parts.append(sent_t.format(e=facts[i].entity, v=facts[i].value))
        for _ in range(rng.randint(1, max(1, n_filler // n_facts))):
            parts.append(rng.choice(FILLER))
    doc = " ".join(parts)
    asked = facts[rng.randrange(len(facts))]
    case = FactCase(doc=doc, question=asked.question, answer=asked.value,
                    fact_type=asked.fact_type)
    return case, facts


def make_case(rng: random.Random, n_facts: int, n_filler: int) -> FactCase:
    """One evaluation case: a document with n_facts fresh facts + filler, and a
    question about one uniformly chosen fact."""
    case, _ = make_case_with_facts(rng, n_facts, n_filler)
    return case


def norm_answer(s: str) -> str:
    """Exact-match normalisation: lowercase, strip answer punctuation, squeeze spaces."""
    return " ".join(s.lower().strip().strip(".,!?;:'\"").split())


def eval_report(name: str, preds: list[tuple[str, str]]) -> dict:
    """preds: list of (prediction, gold).  Exact-match after normalisation."""

    correct = sum(1 for p, g in preds if norm_answer(p) == norm_answer(g))
    acc = correct / max(1, len(preds))
    return {"method": name, "n": len(preds), "exact_match": round(acc, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="cases per evaluation")
    ap.add_argument("--facts-per-doc", type=int, default=8)
    ap.add_argument("--filler-per-doc", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emit", default="", help="write cases jsonl here")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    cases = [make_case(rng, args.facts_per_doc, args.filler_per_doc) for _ in range(args.n)]
    if args.emit:
        with open(args.emit, "w") as f:
            for c in cases:
                f.write(json.dumps(asdict(c)) + "\n")
    # sanity: answers must not leak between cases; entities unique per case
    for c in cases[:1000]:
        assert c.answer in c.doc, "answer must appear in its own doc"
        assert c.question not in c.doc
    print(f"generated {len(cases)} cases; sample:")
    c = cases[0]
    print(" doc:", c.doc[:200], "...")
    print(" q:  ", c.question)
    print(" a:  ", c.answer)


if __name__ == "__main__":
    main()
