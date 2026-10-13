"""
Cell-parallel μLattice scan -- the same math, ordered by CELL instead of
by time.

The delta recurrence is sequential *within* a cell but completely independent
*across* cells, so a token-by-token loop pays T sequential steps when only
max_writes_per_cell steps are needed.  This module:

  1. computes the 3x3 neighbourhood (published bump weights phi) for every token;
  2. keys each write by (sequence, z_x, z_y) and assigns it a RANK = how many
     earlier writes landed on the same cell;
  3. runs max_rank+1 rounds; in round r every cell receives exactly one write,
     so the round is one batched gather/matmul/scatter;
  4. schedules each read into the round where its cell reaches the required
     rank, accumulating its phi-weighted contribution.

Cell count is the number of distinct addresses (no dense grid, no cap); the
number of dispatch rounds is max writes per cell, not the sequence length.
Equivalence with mulattice.sparse_scan.SparseLatticeScan is asserted in
tests/test_kernel.py (max abs diff 8.9e-16 in float64, 16 configurations).
NOTE: the rounds counter is max writes per (sequence, cell), which is <= T; if you
see rounds > T the host's argsort/cummax is not honouring `stable=True` (observed
on one MPS run) and the result must not be trusted.
"""

from __future__ import annotations

import torch

from .kernel import MAX_CELLS_PER_TOKEN, phi_2d
from .sparse_scan import compact_ids as _compact_ids


def _encode(coords: torch.Tensor, n_seq: int, size: int = 1 << 20) -> torch.Tensor:
    """coords (B,T,9,2) int64 -> (B,T,9) collision-free int64 keys."""
    z = coords + (size // 2)
    b = torch.arange(n_seq, device=coords.device, dtype=torch.int64).view(n_seq, 1, 1)
    return (b * (size * size)) + z[..., 0] * size + z[..., 1]


class CellParallelScan(torch.nn.Module):
    def __init__(self, d_k: int, d_v: int, n_heads: int, size: int = 1 << 20,
                 weighted_write: bool = True):
        super().__init__()
        self.d_k, self.d_v, self.n_heads, self.size = d_k, d_v, n_heads, size
        self.weighted_write = weighted_write
        self.last_rounds = 0
        self.last_n_cells = 0

    def forward(
        self,
        q_addr: torch.Tensor,        # (B,T,2) query address (read)  -- same order
        k_addr: torch.Tensor,        # (B,T,2) key address  (write)  -- as SparseLatticeScan
        q2: torch.Tensor,            # (B,T,H,d_k)
        kappa: torch.Tensor,         # (B,T,H,d_k)
        v: torch.Tensor,             # (B,T,H,d_v)
        alpha: torch.Tensor,         # (B,T,H,1)
        beta: torch.Tensor,          # (B,T,H,1)
        read_mode: str = "all",      # "all" | "none"
    ):
        """For every token t: write into its 3x3 cells, then read them.

        Semantics are identical to the token-sequential reference; only the
        execution order changes.
        """
        B, T, H = kappa.shape[0], kappa.shape[1], self.n_heads
        dev, dt = kappa.device, kappa.dtype
        M = MAX_CELLS_PER_TOKEN

        wk, zk = phi_2d(k_addr.reshape(-1, 2))
        wq, zq = phi_2d(q_addr.reshape(-1, 2))
        wk = wk.reshape(B, T, M)
        wq = wq.reshape(B, T, M)
        zk = zk.reshape(B, T, M, 2)
        zq = zq.reshape(B, T, M, 2)

        keys_w = _encode(zk, B, self.size).reshape(-1)              # (B*T*M,)
        keys_q = _encode(zq, B, self.size).reshape(-1)
        inv, n_cells = _compact_ids(torch.cat([keys_w, keys_q]))
        self.last_n_cells = int(n_cells)
        inv_w = inv[: keys_w.numel()].reshape(B, T, M)
        inv_q = inv[keys_w.numel():].reshape(B, T, M)
        cells = torch.zeros(n_cells, H, self.d_v, self.d_k, device=dev, dtype=dt)

        # ---- write events, ordered by (cell, time) -------------------------
        cw = inv_w.reshape(-1)                                       # cell id
        tw = (torch.arange(B * T, device=dev).view(B, T, 1)
              .expand(B, T, M).reshape(-1) % T)                      # time within sequence
        order_w = torch.argsort(cw * (T + 1) + tw, stable=True)
        cw_s, tw_s = cw[order_w], tw[order_w]
        first = torch.ones_like(cw_s, dtype=torch.bool)
        first[1:] = cw_s[1:] != cw_s[:-1]
        ar = torch.arange(cw_s.numel(), device=dev)
        run_start = torch.cummax(torch.where(first, ar, torch.zeros_like(ar)), 0).values
        rank = ar - run_start
        max_rank = int(rank.max()) if rank.numel() else -1
        self.last_rounds = max_rank + 1
        # index of the first write of each cell in the sorted write array
        cell_start = torch.full((n_cells,), cw_s.numel(), device=dev, dtype=torch.int64)
        cell_start[cw_s[first]] = ar[first]

        row_w = (order_w // M)                                       # (b,t) row of the write
        kf = kappa.reshape(B * T, H, self.d_k)[row_w]
        vf = v.reshape(B * T, H, self.d_v)[row_w]
        af = alpha.reshape(B * T, H, 1)[row_w]
        bf = beta.reshape(B * T, H, 1)[row_w]
        # per-write bump weight, in the same (cell, time) order as the writes
        wphi = wk.reshape(-1)[order_w].unsqueeze(-1) if self.weighted_write else None

        # ---- read events: rank of the cell at the moment of the read --------
        acc = None
        if read_mode == "all":
            cq = inv_q.reshape(-1)
            tq = tw
            wkeys_sorted = cw * (T + 1) + tw
            wkeys_sorted = wkeys_sorted[order_w]                     # sorted write keys
            rkeys = cq * (T + 1) + tq
            # writes to THIS cell with key <= read key == writes to this cell with
            # time <= t.  Restricting to the cell's contiguous run is essential:
            # a plain searchsorted would also count writes to lower-id cells.
            need = torch.searchsorted(wkeys_sorted, rkeys, right=True) - cell_start[cq]
            need = need.clamp_(min=0)
            phi_q = wq.reshape(-1)
            row_q = torch.arange(B * T, device=dev).repeat_interleave(M)
            # cell index WITHIN a token (0..M-1).  Each (row, cell) pair is unique, so
            # accumulating into a (B*T, M, H, dv) tensor with index_put_ is deterministic
            # -- unlike index_add_ over duplicate rows, whose CUDA accumulation order is
            # unspecified and makes training non-reproducible across runs.
            cell_i = torch.arange(M, device=dev).repeat(B * T)
            acc = torch.zeros(B * T, M, H, self.d_v, device=dev, dtype=dt)
            qflat = q2.reshape(B * T, H, self.d_k)

        # ---- rounds --------------------------------------------------------
        for r in range(max_rank + 1):
            sel = rank == r
            idx = cw_s[sel]
            W = cells.index_select(0, idx)
            a, b, kap = af[sel], bf[sel], kf[sel]
            if wphi is not None:
                b = b * wphi[sel].unsqueeze(-1)
            v_bar = a * (W @ kap.unsqueeze(-1)).squeeze(-1)
            r_vec = vf[sel] - v_bar
            W = a.unsqueeze(-1) * W + b.unsqueeze(-1) * (
                r_vec.unsqueeze(-1) * kap.unsqueeze(-2))
            # FUNCTIONAL (out-of-place) update.  The in-place index_put_ chained across
            # rounds with interleaved index_select reads corrupted the autograd graph:
            # forward matched the token-loop to ~1e-16 but backward diverged by O(1),
            # which broke every fastscan-trained model (caught by
            # tests/test_kernel.py::test_fastscan_gradient_matches_sparse).  Out-of-place
            # index_put creates a proper graph node per round, so the sequential-recurrence
            # gradient flows correctly.  idx is unique within a round (one write per cell
            # per rank), so the scatter semantics are well defined.
            cells = cells.index_put((idx,), W)
            if read_mode == "all":
                take = need == (r + 1)          # state after exactly r+1 writes
                if take.any():
                    c_idx = cq[take]
                    rows = row_q[take]
                    ci = cell_i[take]
                    Wr = cells.index_select(0, c_idx)
                    out = (Wr @ qflat[rows].unsqueeze(-1)).squeeze(-1)
                    acc.index_put_((rows, ci), phi_q[take].unsqueeze(-1).unsqueeze(-1) * out)

        if read_mode == "all":
            # A read whose cells were never written contributes nothing, which is
            # correct -- but if NO read in the batch hits a written cell, `acc`
            # would stay a plain zeros tensor with no graph, and backward() would
            # fail.  This zero-valued term keeps the read path connected.
            acc = acc.sum(dim=1) + qflat.sum() * 0.0
            return acc.reshape(B, T, H, self.d_v), n_cells
        return None, n_cells
