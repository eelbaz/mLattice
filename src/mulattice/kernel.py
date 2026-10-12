"""
μLattice -- bump kernel and 2D lattice weighting.

The bump kernel and 2D lattice weighting:

    b(t)   = cos^2(pi t / 3)   for |t| < 3/2 ,  0 otherwise          [bump, support 1.5]
    phi_z(x) = b(x - z) / sqrt( sum_j b(x - j)^2 )                   [normalized weights]
    D_1(q,k) = sum_z phi_z(q) phi_z(k)                               [1D kernel]
    D(q,k)   = D_1(q_x,k_x) * D_1(q_y,k_y)                           [2D product kernel]

Consequences that the recipe elevates to invariants:

  * |t| < 3/2 means an axis coordinate touches AT MOST THREE integers,
    so a token touches at most 3 x 3 = 9 lattice cells.  Never more.
  * sum_z phi_z(x)^2 == 1 exactly (L2 partition of unity), and the same
    holds for the 2D product weights, so D(q,q) == 1.
  * Cell fan-out is 3^d for lattice dimension d.  d = 2 -> 9.
"""

from __future__ import annotations

import math

import torch

# Support per bump variant.  The BLOG publishes cos^2(pi t / 3) with |t| < 3/2.
# The reference implementation uses
# cos^2(pi t / 2) with |t| < 1 instead: a narrower bump, at most TWO nonzero cells
# per axis (<=4 in 2D), and exactly ONE when the coordinate is integral.
BUMP_SUPPORT_BY_KIND = {"cos2": 1.5, "shelf": 1.5, "cos2_half": 1.0}
BUMP_SUPPORT = 1.5
LATTICE_DIM = 2
MAX_CELLS_PER_TOKEN = 3 ** LATTICE_DIM  # published: 9


def bump(t: torch.Tensor, kind: str = "cos2") -> torch.Tensor:
    """Bump function on |t| < 3/2, else 0.

    Two variants are published (the second one is the one in the released
    reference figure code):

      "cos2"   b(t) = cos^2(pi t / 3)                             [blog equation]
      "cos2_half" b(t) = cos^2(pi t / 2), |t| < 1  [what the released reference C
               actually uses; note the different support]
      "shelf"  b(t) = exp(-t^2/(2 s^2)) - exp(-2.25/(2 s^2)),  s^2 = 0.72
               (the subtractive term makes it exactly 0 at |t| = 3/2)
    """
    if kind == "cos2":
        inside = t.abs() < BUMP_SUPPORT_BY_KIND["cos2"]
        return torch.where(inside, torch.cos(math.pi * t / 3.0) ** 2, torch.zeros_like(t))
    if kind == "cos2_half":
        inside = t.abs() < BUMP_SUPPORT_BY_KIND["cos2_half"]
        return torch.where(inside, torch.cos(math.pi * t / 2.0) ** 2, torch.zeros_like(t))
    if kind == "shelf":
        s2 = 0.72
        b = torch.exp(-(t * t) / (2 * s2)) - math.exp(-2.25 / (2 * s2))
        return torch.where(t.abs() < BUMP_SUPPORT_BY_KIND["shelf"], b, torch.zeros_like(t))
    raise ValueError(f"unknown bump kind {kind!r}")


def touched_cells(x: torch.Tensor, kind: str = "cos2") -> torch.Tensor:
    """Integer cells an axis-coordinate x can touch, shape (..., 3).

    |x - z| < 3/2  <=>  z in (x - 3/2, x + 3/2), an open interval of length 3,
    which contains at most 3 integers: ceil(x - 3/2), ..., floor(x + 3/2).
    Both candidate ends are returned; the kernel value is 0 (not
    merely small) at |t| == 3/2, so no tolerance is required.
    """
    lo = torch.ceil(x - BUMP_SUPPORT_BY_KIND.get(kind, BUMP_SUPPORT)).to(torch.int64)
    return lo.unsqueeze(-1) + torch.arange(3, device=x.device, dtype=torch.int64)


def phi_1d(x: torch.Tensor, kind: str = "cos2") -> tuple[torch.Tensor, torch.Tensor]:
    """Normalized 1D weights. Returns (weights (...,3), cell coords (...,3)).

    phi_z(x) = b(x - z) / sqrt(sum_j b(x - j)^2)
    """
    z = touched_cells(x, kind)                              # (...,3)
    b = bump(x.unsqueeze(-1) - z.to(x.dtype), kind)         # (...,3)
    denom = torch.sqrt((b * b).sum(dim=-1, keepdim=True)).clamp_min(1e-30)
    return b / denom, z


def phi_2d(xy: torch.Tensor, kind: str = "cos2") -> tuple[torch.Tensor, torch.Tensor]:
    """Normalized 2D weights over the 3x3 neighborhood.

    xy : (..., 2) real-valued address (q or k)
    returns weights (..., 3, 3) and integer coords (..., 3, 3, 2)

    Product form: phi_(zx,zy)(x,y) = phi_zx(x) * phi_zy(y), so
    sum over the 9 cells of phi^2 == 1.
    """
    wx, zx = phi_1d(xy[..., 0], kind)                       # (...,3)
    wy, zy = phi_1d(xy[..., 1], kind)
    w = wx.unsqueeze(-1) * wy.unsqueeze(-2)                 # (...,3,3)
    cz = torch.stack(
        torch.broadcast_tensors(zx.unsqueeze(-1), zy.unsqueeze(-2)), dim=-1
    )                                                       # (...,3,3,2)
    return w, cz


def D_1(q: torch.Tensor, k: torch.Tensor, kind: str = "cos2") -> torch.Tensor:
    """Published 1D kernel D_1(q,k) = sum_z phi_z(q) phi_z(k).

    Evaluated in closed form (Zhuoran Yang / random-feature identity for the
    raised-cosine bump) by summing over the union of the two touch sets; the
    form below is the definition, kept literal for auditability.
    """
    wq, zq = phi_1d(q, kind)                                # (...,3), (...,3)
    # Sum runs only over cells where both weights are nonzero.  Every such
    # cell is in touch(q) (3 candidates), and phi_z(k) for an arbitrary z is
    # b(k - z) / norm_k with norm_k taken over touch(k) -- the definition,
    # since b(k - z) = 0 for every z outside touch(k).
    zk = touched_cells(k, kind)                             # (...,3)
    norm_k = (bump(k.unsqueeze(-1) - zk.to(k.dtype), kind) ** 2).sum(-1, keepdim=True).sqrt().clamp_min(1e-30)
    phi_k_at_zq = bump(k.unsqueeze(-1) - zq.to(k.dtype), kind) / norm_k
    return (wq * phi_k_at_zq).sum(dim=-1)


def D_2(q: torch.Tensor, k: torch.Tensor, kind: str = "cos2") -> torch.Tensor:
    """Published 2D kernel: D(q,k) = D_1(qx,kx) * D_1(qy,ky)."""
    return D_1(q[..., 0], k[..., 0], kind) * D_1(q[..., 1], k[..., 1], kind)


def mix_read(cell_values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Mix per-cell reads by the bump weights.

    cell_values : (..., 3, 3, d_v)   read from each touched cell
    weights     : (..., 3, 3)        normalized bump weights phi
    returns       (..., d_v)
    """
    return (weights.unsqueeze(-1) * cell_values).sum(dim=(-3, -2))
