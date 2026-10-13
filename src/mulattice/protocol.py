"""Shared evaluation/training protocol for the capacity, forgetting and real-data goals.

Single source of truth: the acceptance tests (tests/test_capacity.py,
tests/test_forgetting.py, tests/test_realtext.py) and the training scripts
(experiments/train_capacity.py, experiments/train_forgetting.py,
experiments/realdata_v2.py) all build sequences and measure accuracy through
this module, so "train until the test passes" cannot drift from what is tested.

Synthetic task (capacity + forgetting):
    vocab  keys [0, NK)  values [NK, NK+NV)  SEP = NK+NV+1   V = NK+NV+2
    sequence per row (filler f tokens before each pair; f=0 -> dense MQAR):
        [f k1 v1] [f k2 v2] ... [f kP vP]            store pass
        [f ku1 vu1'] ... [f kuM vuM']                update pass (overwrites)
        [q1 SEP] [q2 SEP] ... [qQ SEP]               query pass
    A query on an updated key must return the UPDATED value (last write wins).
    The answer is read at the SEP token of each query (causal).

Real-data task: wikitext-103 mined key->bigram-successor pairs embedded in a
real corpus filler stream (same mining as experiments/eval_realdata.py).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

# ---------------------------------------------------------------------------
# synthetic generator
# ---------------------------------------------------------------------------


@dataclass
class SynthCfg:
    NK: int = 16384          # distinct key tokens available
    NV: int = 16384          # distinct value tokens available
    d_model: int = 64
    n_heads: int = 2
    d_k: int = 32
    d_v: int = 32
    addr_scale: float = 64.0  # (2*64+3)^2 = 17,161 cells for 8,192 keys
    retention_init: float = 9.0

    @property
    def V(self) -> int:
        return self.NK + self.NV + 2

    @property
    def SEP(self) -> int:
        return self.NK + self.NV + 1


def synth_batch(
    B: int,
    n_pairs: int,
    n_queries: int,
    cfg: SynthCfg,
    filler: int = 0,
    overwrite_frac: float = 0.0,
    generator: torch.Generator | None = None,
    qpos: torch.Tensor | None = None,
    updated_idx: torch.Tensor | None = None,
    kw: dict | None = None,
):
    """Build a capacity/forgetting batch. See module docstring for the layout.

    Returns dict with:
      tokens      (B, L) long          the sequence
      ans_pos     (B, Q) long          SEP positions; answer read there
      gold        (B, Q) long          value token id that must be produced
      was_updated (B, Q) bool          query key was overwritten -> updated gold
      key_pos     (B, P) long          first-pass key token positions (spread loss)
      upd_key_pos (B, M) long | None   update-pass key token positions
      n_pairs, n_upd, L
    """
    g = generator
    NK, NV = cfg.NK, cfg.NV
    if n_pairs > NK:
        raise ValueError("n_pairs > key vocab")
    n_upd = int(round(n_pairs * overwrite_frac))
    if n_pairs + n_upd > NV:
        raise ValueError("not enough distinct values")
    dev = "cpu"  # build on CPU; caller moves to device

    # distinct keys per row; `include_keys` are guaranteed members of the stored
    # set (placed FIRST), and the rest are sampled from the complement, so the
    # batch is consistent by construction -- no duplicate stores of the forced
    # keys with conflicting values (the token-surgery approach produced exactly
    # that and wrecked the model: census 9 -> 713 errors, polish_weakkeys v1).
    include_keys = None
    if "include_keys" in (kw or {}):
        include_keys = kw["include_keys"]
        if include_keys.dim() == 1:
            include_keys = include_keys.unsqueeze(0).expand(B, -1)
    if include_keys is not None:
        nk_inc = include_keys.shape[1]
        assert n_pairs >= nk_inc
        rows = []
        for r in range(B):
            rest = torch.randperm(NK, generator=g)
            rest = rest[torch.isin(rest, include_keys[r], invert=True)][: n_pairs - nk_inc]
            rows.append(torch.cat([include_keys[r], rest]))
        keys = torch.stack(rows)
    else:
        keys = torch.stack([torch.randperm(NK, generator=g)[:n_pairs] for _ in range(B)])
    # distinct values: n_pairs originals + n_upd updates, all different in a row
    vals = torch.stack([torch.randperm(NV, generator=g)[: n_pairs + n_upd] for _ in range(B)])
    orig_vals, upd_vals = vals[:, :n_pairs], vals[:, n_pairs:]

    if updated_idx is None:
        # which stored keys get overwritten (row-specific)
        updated_idx = torch.stack([torch.randperm(n_pairs, generator=g)[:n_upd] for _ in range(B)])
    if qpos is None:
        # DISTINCT query keys (sample without replacement).  Sampling with
        # replacement creates duplicate queries at large Q (measured: 9 of 512
        # at Q=512/NK=16,384) -- not part of the published protocol semantics
        # ("every query has a unique answer", mqar.py); distinct sampling is
        # the clean generalization of the published small-Q generator.
        qpos = torch.stack([torch.randperm(n_pairs, generator=g)[:n_queries] for _ in range(B)])
    else:
        if qpos.dim() == 1:
            qpos = qpos.unsqueeze(0).expand(B, -1).contiguous()
        n_queries = qpos.shape[1]      # explicit query set defines Q
    qkeys = torch.gather(keys, 1, qpos)
    upd_mask = torch.zeros(B, n_pairs, dtype=torch.bool)
    upd_mask.scatter_(1, updated_idx, True)
    was_updated = torch.gather(upd_mask, 1, qpos)
    # latest value for every stored key: original, replaced where updated
    latest = orig_vals.clone()
    latest.scatter_(1, updated_idx, upd_vals)
    gold = torch.gather(latest, 1, qpos) + NK

    f = filler
    block = f + 2
    L = (n_pairs + n_upd) * block + 2 * n_queries

    tokens = torch.zeros(B, L, dtype=torch.long)
    # filler ids: uniform from the complement of the row's stored keys
    if f > 0:
        need = f * (n_pairs + n_upd)
        filler_ids = torch.zeros(B, need, dtype=torch.long)
        keyset = torch.zeros(B, NK, dtype=torch.bool)
        keyset.scatter_(1, keys, True)
        done = torch.zeros(B, need, dtype=torch.bool)
        while not bool(done.all()):
            cand = torch.randint(0, NK, (B, need), generator=g)
            ok = ~torch.gather(keyset, 1, cand) & ~done
            filler_ids = torch.where(ok, cand, filler_ids)
            done = done | ok
    # store pass
    for i in range(n_pairs):
        s = i * block
        if f:
            tokens[:, s:s + f] = filler_ids[:, i * f:(i + 1) * f]
        tokens[:, s + f] = keys[:, i]
        tokens[:, s + f + 1] = orig_vals[:, i] + NK
    # update pass: same key, new value, later in the sequence
    upd_tokens = torch.gather(keys, 1, updated_idx)
    for j in range(n_upd):
        s = (n_pairs + j) * block
        if f:
            tokens[:, s:s + f] = filler_ids[:, (n_pairs + j) * f:(n_pairs + j + 1) * f]
        tokens[:, s + f] = upd_tokens[:, j]
        tokens[:, s + f + 1] = upd_vals[:, j] + NK
    # query pass
    S2 = (n_pairs + n_upd) * block
    for q in range(n_queries):
        tokens[:, S2 + 2 * q] = qkeys[:, q]
        tokens[:, S2 + 2 * q + 1] = cfg.SEP

    key_pos = (torch.arange(n_pairs) * block + f).unsqueeze(0).expand(B, -1).contiguous()
    upd_key_pos = ((n_pairs + torch.arange(n_upd)) * block + f).unsqueeze(0).expand(B, -1).contiguous() \
        if n_upd else None
    ans_pos = (S2 + 2 * torch.arange(n_queries) + 1).unsqueeze(0).expand(B, -1).contiguous()
    return {
        "tokens": tokens, "ans_pos": ans_pos, "gold": gold, "was_updated": was_updated,
        "key_pos": key_pos, "upd_key_pos": upd_key_pos, "n_pairs": n_pairs,
        "n_upd": n_upd, "L": L,
    }


# ---------------------------------------------------------------------------
# model helpers
# ---------------------------------------------------------------------------


def build_model(cfg: SynthCfg, n_layers: int = 1):
    from mulattice.models import MQARModel, MixerCfg
    mcfg = MixerCfg(d_model=cfg.d_model, n_heads=cfg.n_heads, d_k=cfg.d_k, d_v=cfg.d_v,
                    addr_scale=cfg.addr_scale, learn_addr_scale=False, bound_address=True,
                    read_tie=False, retention_init=cfg.retention_init)
    return MQARModel(cfg.V, kind="mulattice", n_layers=n_layers, cfg=mcfg)


def logits_at(model, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Forward the trunk, then apply the head ONLY at `positions` (B,Q).

    Avoids materialising (B,T,V) logits; at T=131K with V=32,770 that is 17GB.
    """
    x = model.emb(tokens)
    for blk in model.blocks:
        x = blk(x, read_mode="all")
    x = model.norm(x)
    B = x.shape[0]
    h = x[torch.arange(B, device=x.device).unsqueeze(1), positions]
    return model.head(h)                       # (B,Q,V)


@torch.no_grad()
def eval_batch(model, batch, device) -> dict:
    """Accuracy of `model` on a synth_batch, split by updated/untouched queries."""
    model.eval()
    pred = logits_at(model, batch["tokens"].to(device), batch["ans_pos"].to(device)).argmax(-1).cpu()
    gold = batch["gold"]
    ok = pred == gold
    res = {"acc": ok.float().mean().item(), "n": ok.numel(),
           "correct": int(ok.sum())}
    up = batch["was_updated"]
    if bool(up.any()):
        res["acc_updated"] = ok[up].float().mean().item()
    if bool((~up).any()):
        res["acc_untouched"] = ok[~up].float().mean().item()
    model.train()
    return res


def scan_info(model) -> str:
    parts = []
    for i, blk in enumerate(model.blocks):
        sc = blk.mixer.scan
        parts.append(f"L{i} cells={sc.last_n_cells} rounds={getattr(sc, 'last_rounds', '?')}"
                     f" path={getattr(sc, 'last_path', '?')}")
    return " ".join(parts)


def force_fast_scan(model):
    for blk in model.blocks:
        blk.mixer.scan.crossover = 1e9


def key_addresses(model, tokens, key_pos):
    """Write addresses (B,N,2) of the tokens at `key_pos` (B,N).

    Uses the mixer's LIVE addr_scale (it can be ramped mid-run), not the
    construction-time cfg value."""
    mx = model.blocks[0].mixer
    xc = mx.conv(model.blocks[0].n1(model.emb(tokens)))
    kaddr = mx.cfg.addr_scale * torch.tanh(mx.Wa(xc))
    ar = torch.arange(tokens.shape[0], device=tokens.device).unsqueeze(1)
    return kaddr[ar, key_pos]


def spread_loss(model, tokens, key_pos, cfg: SynthCfg, gate_subsample: int = 2048):
    """Kernel-D gated spread loss on the TRUE key-token write addresses.

    Same definition as mulattice.addressing.gated_spread_loss, but the
    pairwise write-overlap gate is computed on a subsample when N is large
    (it is (B,N,N); at N=8,192 that is 67M entries per row)."""
    addr = key_addresses(model, tokens, key_pos)
    B, N, _ = addr.shape
    addr_gate = addr[:, torch.randperm(N, device=addr.device)[:gate_subsample]] if N > gate_subsample else addr
    return _gated_spread(addr, addr_gate, float(model.blocks[0].mixer.cfg.addr_scale))


def _gated_spread(addr, addr_gate, scale):
    from mulattice.addressing import write_overlap
    from mulattice.kernel import phi_2d
    B, N, _ = addr.shape
    gate = (write_overlap(addr_gate) - 0.05).clamp(0.0, 1.0).detach()
    lam = 0.3 + 0.7 * gate
    w, z = phi_2d(addr)
    size = int(2 * scale + 3)
    off = int(scale) + 1
    gi = (z + off).clamp(0, size - 1)
    idx = (gi[..., 0] * size + gi[..., 1]).reshape(B, -1).to(torch.int64)
    mass = (w / w.sum(dim=(-2, -1), keepdim=True)).reshape(B, -1)
    hist = torch.zeros(B, size * size, device=addr.device, dtype=addr.dtype)
    for b in range(B):
        hist[b].index_add_(0, idx[b], mass[b])
    p = hist / hist.sum(dim=1, keepdim=True).clamp_min(1e-9)
    entropy = -(p * (p + 1e-12).log()).sum(dim=1)
    return (lam * (-entropy)).mean()


# ---------------------------------------------------------------------------
# vocab surgery: extend the proven 1024-key model to the full key vocabulary
# ---------------------------------------------------------------------------


def extend_from_checkpoint(new_model, ckpt_path: str, NK: int = 16384, NV: int = 16384,
                           old_NK: int = 1024, old_NV: int = 1024,
                           noise: float = 0.05, seed: int = 0):
    """Copy the proven model's mixer into `new_model`; extend emb/head rows.

    New key/value ids are initialised from a donor id (j mod 1024) plus small
    noise, so fresh ids start inside the working recall circuit instead of as
    random noise -- extending a working vocabulary rather than re-bootstrapping.
    From-scratch training at V>=16K historically never escaped the chance plateau
    (logs/vocab8192.log, logs/vocabcur.log).
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    old = ck["model"]
    new = {k: (v.clone() if k not in ("emb.weight", "head.weight") else None)
           for k, v in new_model.state_dict().items()}
    # every non-emb/head parameter has identical shape -> straight copy
    for k in new:
        if new[k] is None:
            continue
        if k not in old or old[k].shape != new[k].shape:
            raise KeyError(f"cannot warm-start: parameter {k} missing/shape-mismatched")
        new[k] = old[k].clone()
    g = torch.Generator().manual_seed(seed)

    def build_rows(rows_old, std, off_new, span_new, off_old, span_old):
        out = torch.empty(span_new, rows_old.shape[1])
        donors = torch.arange(span_new) % span_old
        out[:] = rows_old[off_old + donors]
        out += torch.randn(out.shape, generator=g) * (noise * std)
        return out

    for pname in ("emb.weight", "head.weight"):
        rows_old, rows_new = old[pname], new_model.state_dict()[pname]
        std = rows_old.std().item()
        # keys: first old_NK ids keep their trained rows, the rest inherit donors
        rows_new[:old_NK] = rows_old[:old_NK]
        rows_new[old_NK:NK] = build_rows(rows_old, std, old_NK, NK - old_NK, 0, old_NK)
        # values (offset NK in the new layout, old_NK in the old one)
        rows_new[NK:NK + old_NV] = rows_old[old_NK:old_NK + old_NV]
        rows_new[NK + old_NV:NK + NV] = build_rows(rows_old, std, 0, NV - old_NV, old_NK, old_NV)
        # unused id + SEP
        rows_new[NK + NV] = rows_old[old_NK + old_NV]
        rows_new[NK + NV + 1] = rows_old[old_NK + old_NV + 1]
        new[pname] = rows_new
    new_model.load_state_dict(new)
    return new_model


# ---------------------------------------------------------------------------
# real-data protocol (same mining as experiments/eval_realdata.py)
# ---------------------------------------------------------------------------


class RealData:
    def __init__(self, corpus_path: str, VS: int = 4094):
        from collections import Counter
        self.VS = VS
        words = open(corpus_path, encoding="utf-8", errors="replace").read().lower().split()
        cnt = Counter(words)
        vocab = [w for w, _ in cnt.most_common(VS - 1)]
        wid = {w: i for i, w in enumerate(vocab)}
        big = Counter()
        for a, b in zip(words, words[1:]):
            if a in wid and b in wid:
                big[(wid[a], wid[b])] += 1
        succ = {}
        for (a, b), c in big.items():
            if c >= 4 and (a not in succ or succ[a][1] < c):
                succ[a] = (b, c)
        kv = {}
        ranked = sorted(succ.keys(), key=lambda k: -cnt[vocab[k]])
        lo, hi = int(len(ranked) * 0.02), int(len(ranked) * 0.95)
        for k in ranked[lo:hi]:
            v = succ[k][0]
            if v != k and cnt[vocab[v]] >= 10:
                kv[k] = v
        self.KVLIST = torch.tensor(sorted(kv.keys()))
        self.KVARR = torch.tensor([kv[int(k)] for k in self.KVLIST])
        # Zipfian position weights (cnt^0.75, as in realdata256k.py sample_pairs):
        # concentrating on frequent keys is what let the old run transition in
        # <400 steps; uniform sampling left rare-key embeddings gradient-starved
        # and the model stuck at 0.35 (measured 2026-11-02, realdata_v2 first run).
        w = torch.tensor([float(cnt[vocab[int(k)]]) ** 0.75 for k in self.KVLIST])
        self.KEY_W = w / w.sum()
        self.FILLER = torch.tensor([wid[w] for w in words if w in wid], dtype=torch.long)
        self.KVOCAB = VS
        self.SEP = 2 * VS
        self.V = 2 * VS + 2
        self.n_pairs = len(self.KVLIST)

    def sample_positions(self, B: int, P: int, g: torch.Generator, zipf: bool) -> torch.Tensor:
        """(B,P) distinct KVLIST positions; uniform or Zipf-weighted per row."""
        if not zipf:
            return torch.stack([torch.randperm(self.n_pairs, generator=g)[:P] for _ in range(B)])
        rows = []
        for _ in range(B):
            cand = torch.multinomial(self.KEY_W, 2 * P, replacement=True, generator=g).tolist()
            seen, out, j = set(), [], 0
            while len(out) < P and j < len(cand):
                k = cand[j]; j += 1
                if k not in seen:
                    seen.add(k); out.append(k)
            if len(out) < P:                      # exhausted weighted draws: fill uniformly
                rest = [i for i in torch.randperm(self.n_pairs, generator=g).tolist()
                        if i not in seen][: P - len(out)]
                out += rest
            rows.append(out)
        return torch.tensor(rows)

    def batch(self, P: int, filler: int, Q: int, seed: int, B: int = 1,
              overwrite_frac: float = 0.0, qpos_idx: torch.Tensor | None = None,
              zipf: bool = False):
        """Real-data store/query batch. qpos_idx: (Q,) indices into the P stored
        pairs (allows depth-stratified query sets); overwrite re-presents chosen
        keys with a DIFFERENT value token (another pair's value)."""
        g = torch.Generator().manual_seed(seed)
        P = min(P, self.n_pairs)
        perms = self.sample_positions(B, P, g, zipf)
        keys = self.KVLIST[perms]                       # (B,P) key token ids
        vals = self.KVARR[perms]                        # (B,P) value token ids
        n_upd = int(round(P * overwrite_frac))
        upd_idx = None
        upd_vals = None
        if n_upd:
            upd_idx = torch.stack([torch.randperm(P, generator=g)[:n_upd] for _ in range(B)])
            # update value = another pair's value; redraw the rare self-collisions
            shift = torch.stack([torch.randperm(P, generator=g)[:n_upd] for _ in range(B)])
            upd_vals = torch.gather(vals, 1, shift)
            same = upd_vals == torch.gather(vals, 1, upd_idx)
            while bool(same.any()):
                # rare: key's successor equals donor's successor -> bump by one token id
                flat = same.reshape(-1)
                upd_vals.reshape(-1)[flat] = (upd_vals.reshape(-1)[flat] + 1) % self.VS
                same = upd_vals == torch.gather(vals, 1, upd_idx)
        if qpos_idx is None:
            qpos_idx = torch.stack([torch.randperm(P, generator=g)[:Q] for _ in range(B)])
        elif qpos_idx.dim() == 1:
            qpos_idx = qpos_idx.unsqueeze(0).expand(B, -1).contiguous()
        Q = qpos_idx.shape[1]          # the actual query count (clamped to P)
        qkeys = torch.gather(keys, 1, qpos_idx)

        f = filler
        block = f + 2
        L = (P + n_upd) * block + 2 * Q
        tokens = torch.zeros(B, L, dtype=torch.long)
        fptr = torch.randint(0, max(1, len(self.FILLER) - f * (P + n_upd) - 1), (1,),
                             generator=g).item()
        for i in range(P):
            s = i * block
            if f:
                tokens[:, s:s + f] = self.FILLER[fptr:fptr + f]
                fptr += f
            tokens[:, s + f] = keys[:, i]
            tokens[:, s + f + 1] = vals[:, i] + self.KVOCAB
        if n_upd:
            upd_keys = torch.gather(keys, 1, upd_idx)
            for j in range(n_upd):
                s = (P + j) * block
                if f:
                    tokens[:, s:s + f] = self.FILLER[fptr:fptr + f]
                    fptr += f
                tokens[:, s + f] = upd_keys[:, j]
                tokens[:, s + f + 1] = upd_vals[:, j] + self.KVOCAB
        S2 = (P + n_upd) * block
        latest = vals.clone()
        if n_upd:
            latest.scatter_(1, upd_idx, upd_vals)
        gold = torch.gather(latest, 1, qpos_idx) + self.KVOCAB
        for q in range(Q):
            tokens[:, S2 + 2 * q] = qkeys[:, q]
            tokens[:, S2 + 2 * q + 1] = self.SEP
        key_pos = (torch.arange(P) * block + f).unsqueeze(0).expand(B, -1).contiguous()
        upd_key_pos = ((P + torch.arange(n_upd)) * block + f).unsqueeze(0).expand(B, -1).contiguous() \
            if n_upd else None
        # ANSWER AT THE QUERY-KEY POSITION (even offsets), matching the original
        # realdata256k.py training and eval_realdata.py: `labels[L-2*(n-j)]` is the
        # query-key token, not the SEP after it.  (The SYNTHETIC task answers at
        # the SEP -- mqar.py `ans_pos = arange(1, 2*n_queries, 2)` -- two different
        # conventions, both from the previous session; the warm checkpoint knows
        # the query-key one.  Measured: same tokens score 0.875 under the key
        # convention and 0.000 under the SEP convention.)
        ans_pos = (S2 + 2 * torch.arange(Q)).unsqueeze(0).expand(B, -1).contiguous()
        was_upd = torch.zeros(B, Q, dtype=torch.bool)
        if n_upd:
            m = torch.zeros(B, P, dtype=torch.bool)
            m.scatter_(1, upd_idx, True)
            was_upd = torch.gather(m, 1, qpos_idx)
        return {"tokens": tokens, "ans_pos": ans_pos, "gold": gold, "was_updated": was_upd,
                "key_pos": key_pos, "upd_key_pos": upd_key_pos, "n_pairs": P, "n_upd": n_upd,
                "L": L}


def stratified_qpos(P: int, Q: int, strata: int = 4, seed: int = 0):
    """Query indices stratified over store depth: each stratum gets Q/strata
    queries drawn from pairs stored at that depth fraction of the sequence."""
    g = torch.Generator().manual_seed(seed)
    per = Q // strata
    out = []
    bounds = torch.linspace(0, P, strata + 1).long()
    for s in range(strata):
        lo, hi = int(bounds[s]), int(bounds[s + 1])
        out.append(lo + torch.randperm(hi - lo, generator=g)[:per])
    return torch.cat(out)
