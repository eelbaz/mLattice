"""
μLattice -- growing 2D lattice of fixed-size recurrent cells.

Published (Percepta blogs):
  * the lattice is a 2D grid of cells; a token writes/reads only its 3x3
    neighborhood; routing (2D address) is separated from content;
  * each cell is one DeltaNet state per head, allocated on first write,
    updated by later keys that land on it, all cells sharing one projection
    and one update rule;
  * "work per token does not grow" -- access cost is constant in stored
    information.

CHOICE (not specified by the blogs, required to implement):
  * the concrete gated delta rule (Yang et al. 2024) -- gated delta product:
        v_bar = alpha * W_c kappa
        r     = v - v_bar
        W_c  <- alpha * W_c + beta * r (x) kappa
  * storage as a hash map keyed by integer lattice coordinate (z_x, z_y);
    dense grids are forbidden by the recipe (memory must grow with distinct
    addresses only).

Layout note: W_c in R^{d_v x d_k} so that read(W_c, q) = W_c q in R^{d_v}.
The recipe writes "d_k x d_v"; the transpose is fixed here for consistency
with the read/write equations it gives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .kernel import MAX_CELLS_PER_TOKEN, phi_2d

__all__ = ["LatticeMemory", "LatticeStats"]


@dataclass
class LatticeStats:
    """Bookkeeping for the invariants the recipe says must hold."""

    writes: int = 0
    reads: int = 0
    max_fanout: int = 0
    fanout_hist: dict = field(default_factory=dict)
    touched_cell_ops: int = 0
    stored_tokens: int = 0
    reuses: int = 0

    def record_fanout(self, n: int) -> None:
        self.max_fanout = max(self.max_fanout, n)
        self.fanout_hist[n] = self.fanout_hist.get(n, 0) + 1

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["fanout_hist"] = {int(k): int(v) for k, v in self.fanout_hist.items()}
        return d


class LatticeMemory(torch.nn.Module):
    """Growing hash-map lattice of gated-DeltaNet cells.

    Parameters
    ----------
    d_k : content key/query dim inside a cell
    d_v : value dim inside a cell
    n_heads : independent cell states per lattice coordinate
    device, dtype : where cell tensors live
    """

    def __init__(
        self,
        d_k: int = 64,
        d_v: int = 128,
        n_heads: int = 1,
        device=None,
        dtype=torch.float32,
        weighted_write: bool = True,
    ):
        super().__init__()
        self.d_k, self.d_v, self.n_heads = d_k, d_v, n_heads
        self.device, self.dtype = device, dtype
        self.weighted_write = weighted_write
        self.cells: dict[tuple[int, int], torch.Tensor] = {}
        self.stats = LatticeStats()

    # ---------------------------------------------------------------- basics
    def __len__(self) -> int:
        """Number of allocated cells == size of the growing memory."""
        return len(self.cells)

    def to_device(self, device) -> "LatticeMemory":
        """Move every stored cell to `device` (the hash map is not a submodule)."""
        self.device = device
        for k, t in self.cells.items():
            self.cells[k] = t.to(device)
        return self

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.cells.values())

    def _alloc(self, z: tuple[int, int]) -> torch.Tensor:
        """Allocate a cell on first write (published: allocated on first write)."""
        t = torch.zeros(self.n_heads, self.d_v, self.d_k, device=self.device, dtype=self.dtype)
        self.cells[z] = t
        return t

    # ------------------------------------------------------------ read/write
    @torch.no_grad()
    def write(
        self,
        addr: torch.Tensor,      # (..., 2) real address
        kappa: torch.Tensor,     # (..., H, d_k) content key
        v: torch.Tensor,         # (..., H, d_v) value
        alpha: torch.Tensor,     # (..., H, 1) decay  in (0,1)
        beta: torch.Tensor,      # (..., H, 1) write gate in (0,1)
    ) -> None:
        """Delta-rule write into the 3x3 neighborhood of each address."""
        phi, zs = phi_2d(addr)                     # (...,3,3), (...,3,3,2)
        lead = phi.shape[:-2]
        coords = zs.reshape(-1, MAX_CELLS_PER_TOKEN, 2).cpu().tolist()
        flat_phi = phi.reshape(-1, MAX_CELLS_PER_TOKEN)
        kf = kappa.reshape(-1, self.n_heads, self.d_k)
        vf = v.reshape(-1, self.n_heads, self.d_v)
        af = alpha.reshape(-1, self.n_heads, 1)
        bf = beta.reshape(-1, self.n_heads, 1)

        for i, row in enumerate(coords):
            a, b, kap, val = af[i], bf[i], kf[i], vf[i]
            for j, z in enumerate(row):
                key = (int(z[0]), int(z[1]))
                w = self.cells.get(key)
                if w is None:
                    w = self._alloc(key)
                else:
                    self.stats.reuses += 1
                v_bar = a * (w @ kap.unsqueeze(-1)).squeeze(-1)      # (H, d_v)
                r = val - v_bar
                gate = b * flat_phi[i, j] if self.weighted_write else b
                w.mul_(a.unsqueeze(-1))
                w.add_(gate.unsqueeze(-1) * (r.unsqueeze(-1) * kap.unsqueeze(-2)))
                self.stats.stored_tokens += 1
        self.stats.writes += 1
        # real fan-out: number of cells the token actually wrote (phi > 0)
        live = int((flat_phi.abs() > 0).sum(-1).max())
        self.stats.record_fanout(live)

    # ------------------------------------------- direct (single-cell) access
    # The bump read above is an INTERPOLATING read: it mixes the 3x3
    # neighbourhood, so it cannot return one stored payload exactly.  A reserved
    # region that must reproduce source bytes needs addressing of one cell, which
    # is what the published "a later turn reads that region by the same address"
    # describes.  Both paths share the same cell state and update rule.
    @torch.no_grad()
    def write_cell(self, z: tuple[int, int], kappa, v, alpha, beta) -> None:
        kappa = kappa.reshape(self.n_heads, self.d_k)
        v = v.reshape(self.n_heads, self.d_v)
        alpha = alpha.reshape(self.n_heads, 1)
        beta = beta.reshape(self.n_heads, 1)
        w = self.cells.get(z)
        if w is None:
            w = self._alloc(z)
        else:
            self.stats.reuses += 1
        v_bar = alpha * (w @ kappa.unsqueeze(-1)).squeeze(-1)
        r = v - v_bar
        w.mul_(alpha.unsqueeze(-1))
        w.add_(beta.unsqueeze(-1) * (r.unsqueeze(-1) * kappa.unsqueeze(-2)))
        self.stats.stored_tokens += 1
        self.stats.writes += 1
        self.stats.record_fanout(1)

    @torch.no_grad()
    def read_cell(self, z: tuple[int, int], q2):
        q2 = q2.reshape(self.n_heads, self.d_k)
        w = self.cells.get(z)
        if w is None:
            return torch.zeros(self.n_heads, self.d_v, device=q2.device, dtype=q2.dtype)
        self.stats.reads += 1
        return (w @ q2.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def read(self, addr: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Read: per-cell W_c q2, mixed across the 3x3 neighborhood by phi."""
        phi, zs = phi_2d(addr)
        lead = phi.shape[:-2]
        flat_z = zs.reshape(-1, MAX_CELLS_PER_TOKEN, 2).cpu().tolist()
        flat_phi = phi.reshape(-1, MAX_CELLS_PER_TOKEN).cpu()
        qf = q2.reshape(-1, self.n_heads, self.d_k)
        out = torch.zeros(qf.shape[0], self.n_heads, self.d_v, device=qf.device, dtype=qf.dtype)
        for i, row in enumerate(flat_z):
            acc = torch.zeros(self.n_heads, self.d_v, device=qf.device, dtype=qf.dtype)
            for j, z in enumerate(row):
                w = self.cells.get((int(z[0]), int(z[1])))
                if w is None:            # published: empty cells contribute nothing
                    continue
                acc = acc + flat_phi[i, j] * (w @ qf[i].unsqueeze(-1)).squeeze(-1)
            out[i] = acc
        self.stats.reads += 1
        return out.reshape(*lead, self.n_heads, self.d_v)

    # ------------------------------------------------------------ persistence
    def state_dict_lattice(self) -> dict:
        return {
            "d_k": self.d_k,
            "d_v": self.d_v,
            "n_heads": self.n_heads,
            "cells": {f"{z[0]},{z[1]}": t.detach().cpu() for z, t in self.cells.items()},
            "stats": self.stats.as_dict(),
        }

    def load_state_dict_lattice(self, sd: dict) -> None:
        assert sd["d_k"] == self.d_k and sd["d_v"] == self.d_v and sd["n_heads"] == self.n_heads
        self.cells = {
            tuple(int(x) for x in k.split(",")): t.to(self.device, self.dtype)
            for k, t in sd["cells"].items()
        }

    def save(self, path: str) -> None:
        torch.save(self.state_dict_lattice(), path)

    def load(self, path: str) -> None:
        self.load_state_dict_lattice(torch.load(path, map_location="cpu", weights_only=False))
        if self.device is not None:
            self.to_device(self.device)
