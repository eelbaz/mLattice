"""
Synthetic mixers for the published MQAR evidence.

Structure follows the described synthetic setup: a tiny transformer
(d_model = 64, 2 heads, < 100K matrix parameters of "intelligence") whose
sequence mixer is swapped between

  * μLattice    -- growing 2D lattice of gated-DeltaNet cells
  * Attention   -- causal softmax attention, no positional encodings
  * GatedDeltaNet -- ONE fixed-size state per head (cell count capped at 1:
                   the recipe's own failure rule calls this Gated DeltaNet,
                   not μLattice)

Published for the synthetic mixer: a short conv1d over q,k,v before the mixer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .autoscan import AdaptiveScan
from .sparse_scan import SparseLatticeScan


class ShortConv(nn.Module):
    """Depthwise causal conv1d of width 4 (published: short conv1d on q,k,v)."""

    def __init__(self, d: int, width: int = 4):
        super().__init__()
        self.width = width
        self.conv = nn.Conv1d(d, d, width, groups=d)
        # std = 1/sqrt(width) keeps the conv output at unit scale.  A 0.02 init
        # (the usual "small output layer" default) attenuates the signal ~50x, and
        # because the address head reads the POST-conv activations that
        # collapsed the address spread: 1.15 cells/token instead of ~9.
        nn.init.normal_(self.conv.weight, std=1.0 / math.sqrt(width))
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):                     # (B,T,d)
        y = F.pad(x.transpose(1, 2), (self.width - 1, 0))
        return self.conv(y).transpose(1, 2)


@dataclass
class MixerCfg:
    d_model: int = 64
    n_heads: int = 2
    d_k: int = 32
    d_v: int = 32
    learn_addr_scale: bool = False  # False = fixed scale (PROVEN to bootstrap; the learnable
                                    # scale shrinks and collapses the lattice)
    bound_address: bool = True      # True = tanh-bounded address.  Measured: the unbounded
                                    # linear address (False) breaks routing -- key/query
                                    # addresses drift apart and held-out stays 0.000.
    addr_scale: float = 8.0     # fixed address scale (tanh radius)
    read_tie: bool = False       # addr_q = addr_k.detach() + Waq_delta(xc): the read is
                                 # anchored to the cell the same token writes to (it cannot
                                 # drift off the answer); the delta starts at ZERO so the
                                 # read begins exactly on the write cell.
    retention_init: float | None = None
                                # bias added to the alpha-gate so retention starts near 1.
                                # None = legacy (alpha~0.5), which KILLS credit beyond ~30
                                # steps (tests/test_kernel.py::test_credit_decay_...);
                                # 9.0 -> alpha0~0.999, credit survives 130+ steps.
    conv: bool = True


class MuLatticeMixer(nn.Module):
    def __init__(self, cfg: MixerCfg):
        super().__init__()
        d, H, dk, dv = cfg.d_model, cfg.n_heads, cfg.d_k, cfg.d_v
        self.cfg = cfg
        self.conv = ShortConv(d) if cfg.conv else nn.Identity()
        # (q1, k1) SPLIT -- required by the blog and by the released C, and missing
        # from every earlier version of this file:
        #   "Keys write to a small neighborhood around their addresses, and queries
        #    read from a neighborhood around theirs."
        # reference: phi(p[2h],..) is the WRITE address, phi(p[2H+2h],..) the READ
        # address -- separate projections.  Passing one address for both forces a
        # token's read and write to the same cell, so at the answer position the
        # separator overwrites the very value it is about to retrieve.
        # NOTE: the address is SHARED across heads -- each cell holds one d_k x d_v
        # DeltaNet state PER HEAD, as the blog states, so the cell (not the head) is the
        # routing unit.
        self.Wa = nn.Linear(d, 2)      # k1: write address
        self.Waq_addr = nn.Linear(d, 2)  # q1: read address (untied path)
        if cfg.read_tie:
            self.Waq_delta = nn.Linear(d, 2)
            nn.init.zeros_(self.Waq_delta.weight)
            nn.init.zeros_(self.Waq_delta.bias)
        self.Wq = nn.Linear(d, H * dk)
        self.Wk = nn.Linear(d, H * dk)
        self.Wv = nn.Linear(d, H * dv)
        self.Wg = nn.Linear(d, 2 * H)               # alpha, beta logits
        if cfg.retention_init is not None:
            # alpha for head h is Wg output feature h (view(B,T,2,H): gate*H + head);
            # biasing only the alpha features leaves beta at its default init.
            with torch.no_grad():
                self.Wg.bias[0:H] += cfg.retention_init
        self.Wo = nn.Linear(H * dv, d)
        # NOT zero-init.  With empty cells the read is zero, so a zero gate makes
        # d(loss)/d(gate) = 0 and every mixer parameter is multiplied by a zero
        # gate -- a deadlock: measured gates stuck at 0.002 after 1,200 steps and
        # accuracy pinned at chance forever.  A 1.0 init lets the mixer learn.
        self.out_gate = nn.Parameter(torch.ones(1))
        self.scan = AdaptiveScan(dk, dv, H)
        nn.init.normal_(self.Wa.weight, std=1.0 / (cfg.d_model ** 0.5))
        # CHOICE, and load-bearing: the address map.  A tanh-bounded address
        # (addr = scale*tanh(Wh)) confines coordinates to [-(scale+1), scale+1],
        # i.e. a hard ceiling of (2*scale+3)^2 cells per sequence, and at init the
        # addresses barely spread (+-1.3), so the partition is chaotic: CPU and MPS
        # landed on 258 vs 19 cells for identical inputs, and the collapsed case
        # overloads cells until gradients go inf.  A linear address with a learned
        # scale is unbounded and spreads from step 0.
        self.addr_log_scale = nn.Parameter(
            torch.full((2,), math.log(math.expm1(cfg.addr_scale))))
        self.q_scale = nn.Parameter(torch.zeros(H))   # per-head softplus scale on q2
        self.write_mask = None      # optional (B,T,1,1) gate on writes, for diagnostics
        # Address source: POST-conv (context-dependent).  A pre-conv (token-identity)
        # address was tested and REJECTED: the value token writes at its own cell, not
        # the key's, breaking the value->key-cell induction (measured 0.000 held-out).

    def forward(self, x, read_mode="all"):
        B, T, _ = x.shape
        H, dk, dv = self.cfg.n_heads, self.cfg.d_k, self.cfg.d_v
        xc = self.conv(x)
        if self.cfg.learn_addr_scale:
            sc = F.softplus(self.addr_log_scale)
        else:
            sc = self.cfg.addr_scale
        # (q1, k1) SPLIT: the write address k1 and the read address q1 are separate
        # projections, as in the released C (phi(p[2h]) writes, phi(p[2H+2h]) reads).
        # UNBOUNDED linear address (released C and the blog's "infinitely growing
        # memory"): a tanh cap confines every key to (2*scale+1)^2 cells and makes
        # the capacity claim impossible.  The scale is learned (softplus), so it can
        # grow when more distinct keys need more distinct cells.
        if self.cfg.bound_address:
            addr_k = sc * torch.tanh(self.Wa(xc))
            addr_q = sc * torch.tanh(self.Waq_addr(xc)) if not self.cfg.read_tie \
                else addr_k.detach() + self.Waq_delta(xc)
        else:
            addr_k = self.Wa(xc) * sc
            addr_q = self.Waq_addr(xc) * sc if not self.cfg.read_tie \
                else addr_k.detach() + self.Waq_delta(xc)
        # CHOICE: L2-normalise the content key/query before the delta rule
        # (standard in gated-delta implementations; keeps the recurrence bounded).
        q2 = F.normalize(self.Wq(xc).view(B, T, H, dk), dim=-1) * F.softplus(self.q_scale).view(1, 1, H, 1)
        kap = F.normalize(self.Wk(xc).view(B, T, H, dk), dim=-1)
        v = self.Wv(xc).view(B, T, H, dv)
        gates = self.Wg(xc).view(B, T, 2, H).permute(0, 1, 3, 2)          # (B,T,H,2)
        alpha = torch.sigmoid(gates[..., 0:1]) * 0.999 + 1e-4
        beta = torch.sigmoid(gates[..., 1:2])
        if self.write_mask is not None:
            beta = beta * self.write_mask
        out, n_cells = self.scan(addr_q, addr_k, q2, kap, v, alpha, beta, read_mode=read_mode)
        self.n_cells = n_cells
        y = self.Wo(out.reshape(B, T, H * dv))
        return x + self.out_gate * y


class AttentionMixer(nn.Module):
    def __init__(self, cfg: MixerCfg):
        super().__init__()
        d, H = cfg.d_model, cfg.n_heads
        self.cfg = cfg
        self.conv = ShortConv(d) if cfg.conv else nn.Identity()
        self.Wqkv = nn.Linear(d, 3 * d)
        self.Wo = nn.Linear(d, d)
        self.out_gate = nn.Parameter(torch.ones(1))
        self.head_dim = d // H

    def forward(self, x, read_mode="all"):
        B, T, _ = x.shape
        q, k, v = self.Wqkv(self.conv(x)).chunk(3, dim=-1)
        q = q.view(B, T, self.cfg.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.cfg.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.cfg.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, -1)
        return x + self.out_gate * self.Wo(y)


class GatedDeltaNetMixer(nn.Module):
    """One state per head, fixed size -- no lattice, cell count capped at 1."""

    def __init__(self, cfg: MixerCfg):
        super().__init__()
        d, H, dk, dv = cfg.d_model, cfg.n_heads, cfg.d_k, cfg.d_v
        self.cfg = cfg
        self.conv = ShortConv(d) if cfg.conv else nn.Identity()
        self.Wq = nn.Linear(d, H * dk)
        self.Wk = nn.Linear(d, H * dk)
        self.Wv = nn.Linear(d, H * dv)
        self.Wg = nn.Linear(d, 2 * H)
        self.Wo = nn.Linear(H * dv, d)
        self.out_gate = nn.Parameter(torch.ones(1))

    def forward(self, x, read_mode="all"):
        B, T, _ = x.shape
        H, dk, dv = self.cfg.n_heads, self.cfg.d_k, self.cfg.d_v
        xc = self.conv(x)
        # unit-norm content keys: without this the delta rule's erase term is
        # ||k||^2 ~ d_k times too strong and the state never accumulates
        q = F.normalize(self.Wq(xc).view(B, T, H, dk), dim=-1)
        k = F.normalize(self.Wk(xc).view(B, T, H, dk), dim=-1)
        v = self.Wv(xc).view(B, T, H, dv)
        g = self.Wg(xc).view(B, T, 2, H).permute(0, 1, 3, 2)
        alpha = torch.sigmoid(g[..., 0:1]) * 0.999 + 1e-4
        beta = torch.sigmoid(g[..., 1:2])
        S = torch.zeros(B, H, dv, dk, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(T):
            v_bar = alpha[:, t] * (S @ k[:, t].unsqueeze(-1)).squeeze(-1)
            r = v[:, t] - v_bar
            S = alpha[:, t].unsqueeze(-1) * S + beta[:, t].unsqueeze(-1) * (
                r.unsqueeze(-1) * k[:, t].unsqueeze(-2))
            outs.append((S @ q[:, t].unsqueeze(-1)).squeeze(-1))
        out = torch.stack(outs, 1)                      # (B,T,H,dv)
        self.n_cells = 1
        return x + self.out_gate * self.Wo(out.reshape(B, T, H * dv))


class SwiGLU(nn.Module):
    def __init__(self, d: int, mult: int = 2):
        super().__init__()
        self.w1 = nn.Linear(d, mult * d, bias=False)
        self.w2 = nn.Linear(d, mult * d, bias=False)
        self.w3 = nn.Linear(mult * d, d, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self, d: int, mixer: nn.Module, mlp_mult: int = 2):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mixer = mixer
        self.n2 = nn.RMSNorm(d)
        self.mlp = SwiGLU(d, mlp_mult)

    def forward(self, x, read_mode="all"):
        x = self.mixer(self.n1(x), read_mode=read_mode)
        return x + self.mlp(self.n2(x))


class MQARModel(nn.Module):
    def __init__(self, vocab: int, kind: str = "mulattice", n_layers: int = 2,
                 cfg: MixerCfg | None = None, mlp_mult: int = 2):
        super().__init__()
        cfg = cfg or MixerCfg()
        self.cfg = cfg
        self.kind = kind
        self.emb = nn.Embedding(vocab, cfg.d_model)
        mixers = {
            "mulattice": MuLatticeMixer,
            "attention": AttentionMixer,
            "gdn": GatedDeltaNetMixer,
        }
        self.blocks = nn.ModuleList(
            [Block(cfg.d_model, mixers[kind](cfg), mlp_mult) for _ in range(n_layers)]
        )
        self.norm = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab, bias=False)
        self.n_cells = 0

    def forward(self, tokens, read_mode="all"):
        x = self.emb(tokens)
        for b in self.blocks:
            x = b(x, read_mode=read_mode)
        return self.head(self.norm(x))

    def mixer_params(self) -> int:
        return sum(p.numel() for n, p in self.named_parameters() if ".mixer." in n)

    def memory_cells(self) -> int:
        return max(getattr(b.mixer, "n_cells", 0) for b in self.blocks)
