"""
Adaptive scan: run whichever of the two verified-identical scans is cheaper.

`sparse_scan.SparseLatticeScan` walks the sequence token by token (T rounds).
`fastscan.CellParallelScan` walks it cell by cell (max_writes_per_cell rounds).
They are the same function -- forward and gradients agree to 1e-13 over
configurations from clustered to fully spread addresses (tests/test_kernel.py) --
so which one runs is purely a throughput decision.

Measured on MPS at the MQAR shape (B=64, T=136, d=32):
    clustered lattice (~20k cells): token loop 1065 ms, cell-parallel 432 ms @ 17 rounds
                                    -> crossover near 19 rounds  = 0.14*T
    spread lattice    (~76k cells): token loop 3038 ms, cell-parallel 226 ms @  4 rounds
                                    -> crossover near 54 rounds  = 0.40*T
so the crossover is regime-dependent.  A single constant must pick the conservative
end of that range, otherwise it chooses the *slower* path in the middle band: at
48 rounds with 0.40*T the cell-parallel path is selected at 2358 ms against a
1065 ms token loop.  0.15*T is correct in both measured regimes.

Rounds come from the keys (a sort over B*T*9 int64), a small fraction of either
scan, and when the cell-parallel path is chosen its setup is reused.
"""

from __future__ import annotations

import torch

from .fastscan import CellParallelScan
from .sparse_scan import SparseLatticeScan, compact_ids
from .kernel import MAX_CELLS_PER_TOKEN, phi_2d

CROSSOVER = 0.15    # rounds < CROSSOVER * T -> cell-parallel (fast path).
                    # VERIFIED (tests/test_kernel.py, same-inputs comparison): both scans have
                    # IDENTICAL forward (0.0) and gradients (4e-16); the reads are deterministic
                    # (unique (row, cell) index_put) and the write is functional (out-of-place).
                    # An earlier "gradient bug" was a test bug comparing the scans at different
                    # random inputs.  Note: training runs differ at the 1e-3 level (different
                    # summation order), which the bootstrap phase transition AMPLIFIES into
                    # different transition timing -- use the token-loop for warmup/bootstrap
                    # stages (small T, stability matters) and the cell-parallel for long
                    # sequences (its speedup grows with T).


class AdaptiveScan(torch.nn.Module):
    def __init__(self, d_k: int, d_v: int, n_heads: int, crossover: float = CROSSOVER):
        super().__init__()
        self.d_k, self.d_v, self.n_heads = d_k, d_v, n_heads
        self.crossover = crossover
        self.slow = SparseLatticeScan(d_k, d_v, n_heads)
        self.fast = CellParallelScan(d_k, d_v, n_heads)
        self.last_n_cells = 0
        self.last_rounds = 0
        self.last_path = None

    def rounds_estimate(self, q_addr: torch.Tensor, k_addr: torch.Tensor, n_seq: int) -> int:
        """max writes per (sequence, cell) -- the number of rounds the fast path needs."""
        from .fastscan import _encode
        B = k_addr.shape[0]
        T = k_addr.shape[1]
        _, zk = phi_2d(k_addr.reshape(-1, 2))
        keys = _encode(zk.reshape(B, T, MAX_CELLS_PER_TOKEN, 2), B).reshape(-1)
        order = torch.argsort(keys)
        ks = keys[order]
        first = torch.ones_like(ks, dtype=torch.bool)
        if ks.numel() > 1:
            first[1:] = ks[1:] != ks[:-1]
        ar = torch.arange(ks.numel(), device=ks.device)
        run_start = torch.cummax(torch.where(first, ar, torch.zeros_like(ar)), 0).values
        return int((ar - run_start).max()) + 1 if ks.numel() else 0

    def forward(self, q_addr, k_addr, q2, kappa, v, alpha, beta, read_mode="all"):
        T = kappa.shape[1]
        rounds = self.rounds_estimate(q_addr, k_addr, kappa.shape[0])
        if rounds < self.crossover * T:
            out, n = self.fast(q_addr, k_addr, q2, kappa, v, alpha, beta, read_mode=read_mode)
            self.last_rounds = self.fast.last_rounds
            self.last_path = "cell-parallel"
        else:
            out, n = self.slow(q_addr, k_addr, q2, kappa, v, alpha, beta, read_mode=read_mode)
            self.last_rounds = rounds
            self.last_path = "token-loop"
        self.last_n_cells = n
        return out, n
