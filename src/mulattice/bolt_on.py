"""Frozen μLattice → soft-prompt pipeline shared by bridge training and evaluation.

The bolt-on stores document facts in the frozen capacity8k lattice as
string-hashed synthetic key/value tokens (the protocol the checkpoint was
trained on), queries the lattice with the asked entity, and hands the 64-d
cell readout at the query's SEP position to the trainable bridge.  Prompt
building and answer extraction live here so the floor, ceiling (RAG) and
test conditions of the evaluation share one exact format.
"""
from __future__ import annotations

import hashlib

import torch

KEY_SPACE = 16384          # key vocabulary of the frozen checkpoint (SynthCfg.NK)
SEP_ID = 2 * KEY_SPACE + 1 # query separator token (SynthCfg.SEP)
PAD_ID = 0                 # right padding AFTER a row's SEP: causally invisible
ANSWER_INSTRUCTION = "Answer with the exact value only.\n\n"
FORMAT_EXAMPLE = "Question: What sound does a clock make?\nAnswer: tick\n\n"
READOUT_DIM = 64           # μLattice d_model


class KeyCollision(Exception):
    """Two entities in one document hashed to the same lattice key id."""


def _hash_int(prefix: str, text: str, space: int) -> int:
    digest = hashlib.blake2b((prefix + text).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % space


def key_id(entity: str) -> int:
    """Deterministic lattice key token for an entity string, in [0, KEY_SPACE)."""
    return _hash_int("e:", entity, KEY_SPACE)


def value_id(value: str) -> int:
    """Deterministic lattice value token for a value string, in [KEY_SPACE, 2*KEY_SPACE)."""
    return _hash_int("v:", value, KEY_SPACE) + KEY_SPACE


class FrozenLattice:
    """The frozen capacity8k μLattice used as the bolt-on's fact store.

    Ingestion and query follow the dense protocol the checkpoint was trained
    on: one store pass of adjacent [key value] pairs, then the query pair
    [query_key SEP]; the readout is taken at the SEP position (where the
    training protocol reads its answer)."""

    def __init__(self, ckpt_path: str, device: str = "cpu"):
        from .protocol import SynthCfg, build_model, force_fast_scan
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = SynthCfg()
        saved_scale = ck.get("proto_cfg", {}).get("addr_scale")
        if saved_scale is not None:
            cfg.addr_scale = float(saved_scale)
        model = build_model(cfg)
        model.load_state_dict(ck["model"])
        self.model = model.to(device).eval()
        self.model.requires_grad_(False)
        force_fast_scan(self.model)
        self.device = device

    def _tokens(self, facts: list[tuple[str, str]], query_entity: str) -> list[int]:
        keys = [key_id(e) for e, _ in facts]
        if len(set(keys)) != len(keys):
            raise KeyCollision("two entities hashed to the same lattice key id")
        toks: list[int] = []
        for (ent, val), k in zip(facts, keys):
            toks += [k, value_id(val)]
        toks += [key_id(query_entity), SEP_ID]
        return toks

    @torch.no_grad()
    def _trunk_hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.model.emb(tokens)
        for blk in self.model.blocks:
            x = blk(x, read_mode="all")
        return self.model.norm(x)

    @torch.no_grad()
    def readout(self, facts: list[tuple[str, str]], query_entity: str) -> tuple[torch.Tensor, int, int]:
        """Readout at the query SEP for one case.

        Returns (readout (1, READOUT_DIM), predicted synthetic value id,
        gold synthetic value id)."""
        toks = self._tokens(facts, query_entity)
        t = torch.tensor([toks], dtype=torch.long, device=self.device)
        h = self._trunk_hidden(t)[:, -1]
        pred = int(self.model.head(h).argmax(-1))
        gold = value_id(dict(facts)[query_entity])
        return h, pred, gold

    @torch.no_grad()
    def readout_batch(self, cases: list[tuple[list[tuple[str, str]], str]]) -> tuple[torch.Tensor, list[int], list[int]]:
        """Batched readouts; rows are right-padded after their SEP (causally
        invisible).  Returns ((B, READOUT_DIM), preds, golds)."""
        toks_list = [self._tokens(facts, qe) for facts, qe in cases]
        golds = [value_id(dict(facts)[qe]) for facts, qe in cases]
        width = max(len(t) for t in toks_list)
        batch = torch.full((len(toks_list), width), PAD_ID, dtype=torch.long,
                           device=self.device)
        for i, t in enumerate(toks_list):
            batch[i, :len(t)] = torch.tensor(t, dtype=torch.long, device=self.device)
        h = self._trunk_hidden(batch)
        pos = torch.tensor([len(t) - 1 for t in toks_list], device=self.device)
        h = h[torch.arange(len(toks_list), device=self.device), pos]
        preds = self.model.head(h).argmax(-1).tolist()
        return h, preds, golds


def build_prompt(question: str) -> str:
    """The no-document prompt every condition shares (floor and test).

    One worked example ("tick") teaches the bare-value format so that
    floor and ceiling are not handicapped by formatting; its value is
    outside every boltoneval generator's output space."""
    return f"{ANSWER_INSTRUCTION}{FORMAT_EXAMPLE}Question: {question}\nAnswer:"


def build_rag_prompt(doc: str, question: str) -> str:
    """The ceiling prompt: the document pasted into the context before the question."""
    return f"{doc}\n\n{build_prompt(question)}"


def extract_prediction(generated: str) -> str:
    """The answer span: the first line of the continuation after 'Answer:'."""
    return generated.split("\n")[0].strip()


@torch.no_grad()
def greedy_decode(lm, emb, prompt_ids: torch.Tensor, soft: torch.Tensor | None,
                  eos_ids: tuple[int, ...], max_new_tokens: int = 12) -> list[int]:
    """Greedy decode from [soft tokens | prompt] fed via embeddings.

    Hand-rolled because `generate(inputs_embeds=...)` is unreliable for the
    hybrid linear-attention Qwen3.5 stack (measured: empty or divergent
    continuations), while teacher-forced `forward(inputs_embeds=...)` is
    exact.  Prefill runs on embeddings; each decode step feeds one token id
    with an explicit cache_position.  Verified bit-identical to the
    input_ids generate path on plain prompts and on real-token soft
    prefixes."""
    device = prompt_ids.device
    x = emb(prompt_ids)
    if soft is not None:
        x = torch.cat([soft.to(x.dtype), x], dim=1)
    total = x.shape[1]
    out = lm(inputs_embeds=x, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    gen = [int(nxt.item())]
    for t in range(1, max_new_tokens):
        out = lm(input_ids=nxt, past_key_values=past,
                 cache_position=torch.tensor([total + t - 1], device=device),
                 use_cache=True)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        tok_id = int(nxt.item())
        if tok_id in eos_ids:
            break
        gen.append(tok_id)
    return gen
