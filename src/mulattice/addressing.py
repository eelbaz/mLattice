"""Address-diversity regularisation for μLattice (experimental).

A mixture-of-experts-style load-balancing term applied to addresses: maximise the entropy of
the soft cell-usage distribution so addresses stay spread instead of collapsing onto a few
cells.

Status: the *collapse* hypothesis that motivated this was later contradicted -- on CUDA a run
with every key in one cell still generalised (0.875 at 256 keys), and the term did not
reproducibly help.  It is kept as a documented experiment, not a recommended part of the mixer.
"""
from __future__ import annotations

import torch


def usage_entropy(addr: torch.Tensor, scale: float, grid: int = 24) -> torch.Tensor:
    """Entropy (nats) of the soft cell-usage distribution of `addr` (..., 2).

    Addresses are normalised into a `grid` x `grid` histogram by differentiable bilinear
    assignment, so the entropy carries gradient back to the address projection.  Maximum is
    log(grid**2); a collapsed map gives near zero.
    """
    x = (addr[..., 0].clamp(-scale, scale) + scale) / (2 * scale)      # -> [0, 1]
    y = (addr[..., 1].clamp(-scale, scale) + scale) / (2 * scale)
    gx = (x * grid).clamp(0.0, float(grid) - 1e-6)                     # [0, grid)
    gy = (y * grid).clamp(0.0, float(grid) - 1e-6)
    ix = gx.floor().long(); iy = gy.floor().long()
    fx = (gx - ix.float()).reshape(-1); fy = (gy - iy.float()).reshape(-1)
    ix1 = (ix + 1).clamp(max=grid - 1); iy1 = (iy + 1).clamp(max=grid - 1)

    p = torch.zeros(grid * grid, device=addr.device, dtype=torch.float32)
    p.index_add_(0, (ix * grid + iy).reshape(-1),     ((1 - fx) * (1 - fy)).float())
    p.index_add_(0, (ix1 * grid + iy).reshape(-1),    (fx * (1 - fy)).float())
    p.index_add_(0, (ix * grid + iy1).reshape(-1),    ((1 - fx) * fy).float())
    p.index_add_(0, (ix1 * grid + iy1).reshape(-1),   (fx * fy).float())
    p = p / p.sum().clamp_min(1e-9)
    return -(p * (p + 1e-12).log()).sum()


def address_diversity_loss(addr: torch.Tensor, scale: float, grid: int = 24) -> torch.Tensor:
    """Minimise this to keep addresses spread: negative entropy of cell usage."""
    return -usage_entropy(addr, scale, grid)


def write_overlap(addr: torch.Tensor) -> torch.Tensor:
    """Mean pairwise published-kernel overlap of write addresses, (B, N, 2) -> (B,).

    D(k_i, k_j) = D1(x) * D1(y) from kernel.py measures exactly what the scan interferes
    on: shared 3x3 cell neighbourhoods.  This is the quantity the collision gate uses --
    NOT a grid histogram of the address box.
    """
    from .kernel import D_1
    B, N, _ = addr.shape
    x = addr[..., 0]
    y = addr[..., 1]
    dx = D_1(x[:, :, None], x[:, None, :])           # (B, N, N)
    dy = D_1(y[:, :, None], y[:, None, :])
    D = dx * dy
    diag = N                                           # D(i,i) == 1 each
    off = (D.sum(dim=(1, 2)) - diag) / max(N * (N - 1), 1)
    return off


def gated_spread_loss(addr: torch.Tensor, scale: float, gate_thresh: float = 0.05) -> torch.Tensor:
    """Collision-gated spread: lambda_t = clamp(overlap - 0.05, 0, 1) times the NEGATIVE
    entropy of centre-cell usage, where usage is the real phi weights on the real integer
    cells (not a histogram grid).  Zero when the keys are already spread; positive only
    while they actually collide."""
    from .kernel import phi_2d
    B, N, _ = addr.shape
    # Floor 0.3 + gated escalation to 1.0.  A pure gate self-disables at TRAIN-time key
    # counts (64 keys spread well enough that overlap < thresh), leaving eval-time packing
    # (256 keys) unaddressed -- measured: gated-only plateaued @256 at 0.66 vs 0.91 baseline.
    gate = (write_overlap(addr) - gate_thresh).clamp(0.0, 1.0).detach()
    lam = 0.3 + 0.7 * gate
    w, z = phi_2d(addr)                                # (B,N,3,3), (B,N,3,3,2) int coords
    size = int(2 * scale + 3)
    off = int(scale) + 1                                # keep integer arithmetic
    gi = (z + off).clamp(0, size - 1)
    idx = (gi[..., 0] * size + gi[..., 1]).reshape(B, -1).to(torch.int64)
    mass = (w / w.sum(dim=(-2, -1), keepdim=True)).reshape(B, -1)   # normalised per key
    hist = torch.zeros(B, size * size, device=addr.device, dtype=addr.dtype)
    idx_f = idx.reshape(B, -1)
    mass_f = mass.reshape(B, -1)
    for b in range(B):                                  # index_add_ needs 1-D indices
        hist[b].index_add_(0, idx_f[b], mass_f[b])
    p = hist / hist.sum(dim=1, keepdim=True).clamp_min(1e-9)
    entropy = -(p * (p + 1e-12).log()).sum(dim=1)
    return (lam * (-entropy)).mean()
