"""
Batched, differentiable μLattice lattice scan.

Same published math as mulattice.lattice.LatticeMemory, but the sequential
recurrence is executed with tensor ops over the whole batch instead of a Python
loop over (token x 9 cells), so the synthetic MQAR mixers can be trained.

Sparsity is preserved: cells are keyed by (sequence, z_x, z_y) via
torch.unique, and only the keys that actually occur are allocated.  No dense
grid over the address bounding box is ever materialised.  Cell count is
unbounded -- it is the number of distinct addresses written.

Semantics identical to the reference implementation:
    phi from the published bump kernel, exactly 9 cells per token,
    v_bar = alpha * W_c kappa ;  r = v - v_bar ;  W_c <- alpha W_c + beta r kappa^T
    read  = sum over the 3x3 neighbourhood of phi_c * (W_c q2), empty cells 0.
"""

from __future__ import annotations

import torch

from .kernel import MAX_CELLS_PER_TOKEN, phi_2d


def compact_ids(keys: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Map arbitrary int64 keys to dense contiguous ids, via sort.

    Uses sort + run detection instead of torch.unique: on MPS, torch.unique corrupts int64
    values outside the int32 range (measured: 4959 distinct keys reported as 2789,
    with returned values like -30933022 instead of 549724880885).  That scrambled
    the cell ids, collapsed the lattice onto a few cells, and made training
    diverge to inf gradients on MPS while CPU was fine.  Sort + run detection is
    exact on every backend.
    """
    if keys.numel() == 0:
        return keys.clone(), 0
    order = torch.argsort(keys)
    ks = keys[order]
    first = torch.ones_like(ks, dtype=torch.bool)
    first[1:] = ks[1:] != ks[:-1]
    ids_sorted = torch.cumsum(first.to(torch.int64), 0) - 1
    ids = torch.empty_like(ids_sorted)
    ids[order] = ids_sorted
    return ids, int(first.sum())


def _encode_keys(coords: torch.Tensor, n_seq: int, size: int = 1 << 20) -> torch.Tensor:
    """coords (B,T,9,2) int64 -> flat int64 keys, collision-free for |z| < size/2."""
    z = coords + (size // 2)
    b = torch.arange(n_seq, device=coords.device, dtype=torch.int64).view(n_seq, 1, 1)
    return (b * (size * size)) + z[..., 0] * size + z[..., 1]


class SparseLatticeScan(torch.nn.Module):
    def __init__(self, d_k: int, d_v: int, n_heads: int, size: int = 1 << 20,
                 weighted_write: bool = True):
        super().__init__()
        self.d_k, self.d_v, self.n_heads, self.size = d_k, d_v, n_heads, size
        self.weighted_write = weighted_write
        self.last_n_cells = 0

    def forward(
        self,
        q_addr: torch.Tensor,     # (B,T,2) real address of the query
        k_addr: torch.Tensor,     # (B,T,2) real address of the key
        q2: torch.Tensor,         # (B,T,H,d_k) content query
        kappa: torch.Tensor,      # (B,T,H,d_k) content key
        v: torch.Tensor,          # (B,T,H,d_v) value
        alpha: torch.Tensor,      # (B,T,H,1) decay in (0,1)
        beta: torch.Tensor,       # (B,T,H,1) write gate in (0,1)
        read_mode: str = "last",  # "last" | "all" | "none"
        causal_read: bool = True,
    ):
        B, T, H = kappa.shape[0], kappa.shape[1], self.n_heads
        dev, dt = kappa.device, kappa.dtype

        wk, zk = phi_2d(k_addr)                                  # (B,T,3,3), (B,T,3,3,2)
        wq, zq = phi_2d(q_addr)
        wk = wk.reshape(B, T, MAX_CELLS_PER_TOKEN)
        wq = wq.reshape(B, T, MAX_CELLS_PER_TOKEN)
        zk = zk.reshape(B, T, MAX_CELLS_PER_TOKEN, 2)
        zq = zq.reshape(B, T, MAX_CELLS_PER_TOKEN, 2)

        need_q = read_mode != "none"
        keys_w = _encode_keys(zk, B, self.size).reshape(-1)
        keys_q = _encode_keys(zq, B, self.size).reshape(-1) if need_q else keys_w[:0]
        all_keys = torch.cat([keys_w, keys_q]) if need_q else keys_w
        inv, n_cells = compact_ids(all_keys)
        self.last_n_cells = int(n_cells)
        inv_w = inv[: keys_w.numel()].reshape(B, T, MAX_CELLS_PER_TOKEN)
        inv_q = (
            inv[keys_w.numel():].reshape(B, T, MAX_CELLS_PER_TOKEN) if need_q else None
        )

        # cells allocated lazily: only distinct keys that occur, zero-initialised
        cells = torch.zeros(n_cells, H, self.d_v, self.d_k, device=dev, dtype=dt)

        outs = []
        for t in range(T):
            idx = inv_w[:, t]                                     # (B,9)
            W = cells.index_select(0, idx.reshape(-1)).reshape(B, MAX_CELLS_PER_TOKEN, H, self.d_v, self.d_k)
            kap = kappa[:, t].unsqueeze(1)                        # (B,1,H,dk)
            a = alpha[:, t].unsqueeze(1)                          # (B,1,H,1)
            b = beta[:, t].unsqueeze(1)
            v_bar = a * (W @ kap.unsqueeze(-1)).squeeze(-1)       # (B,9,H,dv)
            r = v[:, t].unsqueeze(1) - v_bar
            gate = b * wk[:, t].unsqueeze(-1).unsqueeze(-1) if self.weighted_write else b
            W = a.unsqueeze(-1) * W + gate.unsqueeze(-1) * (r.unsqueeze(-1) * kap.unsqueeze(-2))
            cells.index_put_((idx.reshape(-1),), W.reshape(-1, H, self.d_v, self.d_k))

            if read_mode == "all":
                qi = inv_q[:, t] if causal_read else None
                if qi is None:                                    # non-causal: after full seq
                    outs.append(None)
                else:
                    Wq = cells.index_select(0, qi.reshape(-1)).reshape(
                        B, MAX_CELLS_PER_TOKEN, H, self.d_v, self.d_k)
                    rq = (Wq @ q2[:, t].unsqueeze(1).unsqueeze(-1)).squeeze(-1)   # (B,9,H,dv)
                    outs.append((wq[:, t].unsqueeze(-1).unsqueeze(-1) * rq).sum(1))

        if read_mode == "last":
            qi = inv_q[:, T - 1]                                  # (B,9)
            Wq = cells.index_select(0, qi.reshape(-1)).reshape(
                B, MAX_CELLS_PER_TOKEN, H, self.d_v, self.d_k)
            rq = (Wq @ q2[:, T - 1].unsqueeze(1).unsqueeze(-1)).squeeze(-1)
            out = (wq[:, T - 1].unsqueeze(-1).unsqueeze(-1) * rq).sum(1)   # (B,H,dv)
            return out, n_cells
        if read_mode == "all":
            return torch.stack(outs, dim=1), n_cells              # (B,T,H,dv)
        return None, n_cells
