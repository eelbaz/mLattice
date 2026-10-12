#!/usr/bin/env python3
"""
Invariant tests for the μLattice kernel and lattice.

Run:  python3 tests/test_kernel.py
Exits non-zero on the first failed assertion.  No test framework required.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from mulattice.fastscan import CellParallelScan
from mulattice.kernel import (
    BUMP_SUPPORT,
    MAX_CELLS_PER_TOKEN,
    D_1,
    D_2,
    bump,
    phi_1d,
    phi_2d,
    touched_cells,
)
from mulattice.lattice import LatticeMemory
from mulattice.sparse_scan import SparseLatticeScan

FAILED = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILED.append(name)


def test_fanout():
    """'Each key therefore writes and reads exactly 3x3 cells' -- never more."""
    for kind in ("cos2", "shelf"):
        xs = torch.linspace(-12, 12, 400001, dtype=torch.float64)
        w, _ = phi_1d(xs, kind)
        nz = (w.abs() > 0).sum(-1)
        check(f"1D fan-out <= 3 [{kind}]", int(nz.max()) <= 3, f"max={int(nz.max())}")

        # adversarial: exact half-integers, just inside/outside the support
        adv = torch.tensor([-1.5, 1.5, -0.5, 0.5, 2.5, -2.5, 1.4999999, 1.5000001,
                            1e-16, 3.0, -3.0, 0.0, 7.5, -7.5], dtype=torch.float64)
        wa, _ = phi_1d(adv, kind)
        check(f"1D adversarial fan-out <= 3 [{kind}]", int((wa.abs() > 0).sum(-1).max()) <= 3)

        xy = torch.rand(100000, 2, dtype=torch.float64) * 40 - 20
        W, _ = phi_2d(xy, kind)
        f = (W.abs() > 0).sum(dim=(-1, -2))
        check(f"2D fan-out exactly 9 [{kind}]",
              int(f.min()) == 9 and int(f.max()) == 9, f"[{int(f.min())},{int(f.max())}]")


def test_normalisation():
    for kind in ("cos2", "shelf"):
        xy = torch.rand(20000, 2, dtype=torch.float64) * 20 - 10
        W, _ = phi_2d(xy, kind)
        s = (W * W).sum(dim=(-1, -2))
        check(f"sum phi^2 == 1 [{kind}]", float((s - 1).abs().max()) < 1e-12,
              f"max dev {float((s - 1).abs().max()):.2e}")
        q = torch.rand(2000, 2, dtype=torch.float64) * 10 - 5
        check(f"D(q,q) == 1 [{kind}]", float((D_2(q, q, kind) - 1).abs().max()) < 1e-12)
        k = torch.rand(2000, 2, dtype=torch.float64) * 10 - 5
        check(f"D symmetric [{kind}]", float((D_2(q, k, kind) - D_2(k, q, kind)).abs().max()) < 1e-12)
        shift = torch.tensor([13.0, -7.0], dtype=torch.float64)
        check(f"D translation invariant [{kind}]",
              float((D_2(q + shift, k + shift, kind) - D_2(q, k, kind)).abs().max()) < 1e-12)
        check(f"b(+/-1.5) == 0 [{kind}]",
              float(bump(torch.tensor([1.5, -1.5], dtype=torch.float64), kind).abs().max()) == 0.0)


def test_D1_matches_definition():
    """D_1 against the literal sum_z phi_z(q) phi_z(k) over a wide integer range."""
    def brute(q, k, kind):
        z = torch.arange(-60, 61, dtype=torch.float64)
        bq = bump(q.unsqueeze(-1) - z, kind)
        bk = bump(k.unsqueeze(-1) - z, kind)
        bq = bq / (bq ** 2).sum(-1, keepdim=True).sqrt().clamp_min(1e-30)
        bk = bk / (bk ** 2).sum(-1, keepdim=True).sqrt().clamp_min(1e-30)
        return (bq * bk).sum(-1)
    for kind in ("cos2", "shelf"):
        q = torch.rand(200, dtype=torch.float64) * 20 - 10
        k = torch.rand(200, dtype=torch.float64) * 20 - 10
        check(f"D_1 == brute force [{kind}]",
              float((D_1(q, k, kind) - brute(q, k, kind)).abs().max()) < 1e-12)


def test_implementations_agree(scale=6.0, B=2, T=11, H=2, dk=8, dv=6, tag=""):
    """Hash map, batched scan and cell-parallel scan are the same function."""
    torch.manual_seed(0)
    dt = torch.float64
    ka = (torch.rand(B, T, 2, dtype=dt) * 2 - 1) * scale
    qa = (torch.rand(B, T, 2, dtype=dt) * 2 - 1) * scale
    q2 = torch.randn(B, T, H, dk, dtype=dt)
    kap = torch.randn(B, T, H, dk, dtype=dt)
    v = torch.randn(B, T, H, dv, dtype=dt)
    al = torch.rand(B, T, H, 1, dtype=dt) * 0.9 + 0.05
    be = torch.rand(B, T, H, 1, dtype=dt) * 0.8

    ref = SparseLatticeScan(dk, dv, H).to(dt)
    fast = CellParallelScan(dk, dv, H).to(dt)
    o_ref, n_ref = ref(qa, ka, q2, kap, v, al, be, read_mode="all")
    o_fast, n_fast = fast(qa, ka, q2, kap, v, al, be, read_mode="all")
    check(f"batched scan == cell-parallel scan{tag}",
          float((o_ref - o_fast).abs().max()) < 1e-9,
          f"max diff {float((o_ref - o_fast).abs().max()):.2e}, cells {n_ref}/{n_fast}")

    # hash-map reference on the last read position
    outs = []
    for b in range(B):
        mem = LatticeMemory(dk, dv, H, dtype=dt)
        for t in range(T):
            mem.write(ka[b, t], kap[b, t], v[b, t], al[b, t], be[b, t])
        outs.append(mem.read(qa[b, T - 1], q2[b, T - 1]))
    o_hash = torch.stack(outs, 0)
    o_last, _ = SparseLatticeScan(dk, dv, H).to(dt)(qa, ka, q2, kap, v, al, be, read_mode="last")
    check(f"batched scan == hash-map lattice{tag}",
          float((o_hash - o_last).abs().max()) < 1e-12,
          f"max diff {float((o_hash - o_last).abs().max()):.2e}")


def test_growth_and_cost():
    """Cells grow with distinct addresses (uncapped); per-token work stays flat."""
    mem = LatticeMemory(4, 4, 1)
    for i in range(400):
        mem.write(torch.tensor([float(i % 40), float(i // 40)]),
                  torch.ones(1, 4), torch.ones(1, 4),
                  torch.tensor([[0.9]]), torch.tensor([[0.5]]))
    check("lattice grows with distinct addresses", len(mem) > 400 // 2, f"cells={len(mem)}")
    check("fan-out recorded as exactly 9", mem.stats.max_fanout == MAX_CELLS_PER_TOKEN)

    def per_token_ms(n_pairs, dk=32, dv=32, H=2, B=8):
        sc = SparseLatticeScan(dk, dv, H)
        T = 2 * n_pairs + 4
        a = torch.rand(B, T, 2) * 2000 - 1000
        q2 = torch.randn(B, T, H, dk); k = torch.randn(B, T, H, dk); v = torch.randn(B, T, H, dv)
        al = torch.rand(B, T, H, 1) * 0.5 + 0.4; be = torch.rand(B, T, H, 1) * 0.5
        sc(a, a, q2, k, v, al, be, read_mode="last")
        t0 = time.time()
        out, n = sc(a, a, q2, k, v, al, be, read_mode="last")
        return (time.time() - t0) / T / B * 1e6, n

    # min of 3 runs: single-call wall-clock is noisy on a shared GPU
    small, n_small = min(per_token_ms(64) for _ in range(3))
    large, n_large = min(per_token_ms(2048) for _ in range(3))
    ratio = large / small
    # O(1) per token means the ratio should be near 1; 2.0 absorbs scheduler
    # jitter on shared GPUs.  Attention's ratio over the same span is ~32x.
    check("per-token cost flat as memory grows", ratio < 2.0,
          f"{small:.1f} -> {large:.1f} us/token/seq as cells {n_small} -> {n_large} (ratio {ratio:.2f})")


def test_read_gain_and_direct_access():
    """Write weighting decides the read gain.

    The blogs do not say whether a write is weighted by the bump; the released C
    weights it (`w = wx[i]*wy[j]*beta`).  Weighted write + weighted read gives
    sum(phi^2) = 1, i.e. an unbiased round trip; an unweighted write gives
    sum(phi) = 2.  Both are checked, and the default is the released behaviour.
    """
    phi, _ = phi_2d(torch.tensor([[0.0, 0.0]], dtype=torch.float64))
    check("sum(phi) == 2 at an integer address", abs(float(phi.sum()) - 2.0) < 1e-12,
          f"sum={float(phi.sum()):.6f}")
    check("sum(phi^2) == 1", abs(float((phi * phi).sum()) - 1.0) < 1e-12)

    key = torch.zeros(1, 8); key[0, 0] = 1.0
    val = torch.zeros(1, 8); val[0, :4] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    addr = torch.tensor([[0.0, 0.0]])
    unit = (torch.ones(1, 1, 1), torch.ones(1, 1, 1))

    for weighted, gain in ((True, 1.0), (False, 2.0)):
        mem = LatticeMemory(8, 8, 1, weighted_write=weighted)
        mem.write(addr, key, val, *unit)
        mixed = mem.read(addr, key)[0, 0, :4]
        check(f"bump-mixed read gain == {gain:g} (weighted_write={weighted})",
              float((mixed - gain * val[0, :4]).abs().max()) < 1e-5,
              f"{[round(float(x), 4) for x in mixed]}")

    mem2 = LatticeMemory(8, 8, 1)
    mem2.write_cell((0, 0), key, val, torch.ones(1, 1), torch.ones(1, 1))
    direct = mem2.read_cell((0, 0), key)[0, :4]
    check("direct cell read is exact", float((direct - val[0, :4]).abs().max()) == 0.0,
          f"{direct.tolist()}")


def test_write_overlap_and_gated_spread():
    """Collision-gated spread (the corrected objective): mean pairwise kernel-D of write
    addresses gates an entropy term computed on the REAL integer cells (phi weights), not a
    grid histogram.  Packed keys -> high overlap -> gate open; spread keys -> gate closed."""
    import torch
    from mulattice.addressing import write_overlap, gated_spread_loss

    # packed: 64 keys in one cluster -> mean pairwise D ~ 1
    packed = torch.zeros(1, 64, 2) + 0.01 * torch.randn(1, 64, 2)
    ov_pack = write_overlap(packed)
    # spread: 25 keys on a 5x5 grid with spacing 3.5 (inside the scale-8 box) -> D ~ 0
    g = torch.arange(25, dtype=torch.float32)
    spread = torch.stack([(g % 5) * 3.5 - 7.0, (g // 5) * 3.5 - 7.0], dim=-1).unsqueeze(0)
    ov_spr = write_overlap(spread)
    check("packed keys have high kernel-D overlap", float(ov_pack) > 0.8, f"{float(ov_pack):.3f}")
    check("spread keys have ~zero overlap", float(ov_spr) < 0.05, f"{float(ov_spr):.3f}")

    l_pack = gated_spread_loss(packed, scale=8.0)
    l_spr = gated_spread_loss(spread, scale=8.0)
    check("gate open when packed (loss active, < -0.1)", float(l_pack) < -0.1, f"{float(l_pack):.4f}")
    # corrected design: floor 0.3 always active, escalation adds more when packed
    check("floor active when spread (loss nonzero)", abs(float(l_spr)) > 1e-6, f"{float(l_spr):.4f}")
    check("escalation makes packed loss stronger than spread", abs(float(l_pack)) > abs(float(l_spr)),
          f"|{float(l_pack):.3f}| > |{float(l_spr):.3f}|")


def test_read_tie():
    """addr_q = addr_k.detach() + delta: with zero-init delta the read starts EXACTLY on the
    write cell; gradients reach delta but not Wa through the tie."""
    import torch
    from mulattice.models import MQARModel, MixerCfg
    from mulattice.mqar import make_mqar_batch, vocab_size
    torch.manual_seed(0)
    m = MQARModel(2050, kind="mulattice", n_layers=1,
                  cfg=MixerCfg(d_model=64, n_heads=2, d_k=32, d_v=32, addr_scale=8.0,
                               learn_addr_scale=False, bound_address=True, read_tie=True))
    mx = m.blocks[0].mixer
    b = make_mqar_batch(2, 4, 4, 1024, 1024, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        x = m.blocks[0].n1(m.emb(b["tokens"]))
        xc = mx.conv(x)
        ak = 8.0 * torch.tanh(mx.Wa(xc))
        aq = ak.detach() + mx.Waq_delta(xc)
        check("read address starts exactly on the write cell (zero-init tie)",
              torch.equal(aq, ak), f"max|aq-ak| = {float((aq-ak).abs().max()):.1e}")
    out = m(b["tokens"], read_mode="all")
    out.sum().backward()
    check("gradient flows to the tie delta", mx.Waq_delta.weight.grad is not None
          and float(mx.Waq_delta.weight.grad.abs().sum()) > 0,
          "delta grad nonzero")


if __name__ == "__main__":
    print("=" * 78)
    print("μLattice kernel / lattice invariant tests")
    print("=" * 78)
    test_fanout()
    test_normalisation()
    test_D1_matches_definition()
    test_implementations_agree()
    test_write_overlap_and_gated_spread()
    test_read_tie()
    # production config: the address scale the MQAR runs and the graft use now
    test_implementations_agree(scale=256.0, B=2, T=32, H=2, dk=16, dv=16, tag=" [scale 256]")
    test_implementations_agree(scale=64.0, B=1, T=64, H=1, dk=64, dv=128, tag=" [graft shape]")
    test_growth_and_cost()
    test_read_gain_and_direct_access()
    print("=" * 78)
    if FAILED:
        print(f"{len(FAILED)} FAILED: {FAILED}")
        sys.exit(1)
    print("all checks passed")


def test_fastscan_gradient_matches_sparse():
    """RED-REG test: the cell-parallel scan must produce the SAME gradient as the token-loop
    reference.  It does not -- its forward matches to ~1e-16 but its backward differs by O(1)
    (in-place index_put_ across rank rounds corrupts the graph), which made every
    fastscan-trained model wrong and non-reproducible.  This stays failing until fixed."""
    import torch.nn.functional as F
    from mulattice.sparse_scan import SparseLatticeScan
    from mulattice.fastscan import CellParallelScan
    torch.manual_seed(0)
    B, T, H, dk, dv = 2, 16, 2, 8, 8
    def inputs():
        q_addr = torch.randn(B,T,2, dtype=torch.float64)
        k_addr = torch.randn(B,T,2, dtype=torch.float64)
        q2 = F.normalize(torch.randn(B,T,H,dk, dtype=torch.float64), dim=-1)
        kap = F.normalize(torch.randn(B,T,H,dk, dtype=torch.float64), dim=-1)
        v = torch.randn(B,T,H,dv, dtype=torch.float64)
        alpha = torch.rand(B,T,H,1, dtype=torch.float64)*0.5+0.5
        beta = torch.rand(B,T,H,1, dtype=torch.float64)
        return q_addr,k_addr,q2,kap,v,alpha,beta
    def grads(scan, *args):
        for a in args: a.requires_grad_(True); a.grad = None
        o, n = scan(*args, read_mode="all"); o.sum().backward()
        return [a.grad for a in args]
    sp = SparseLatticeScan(dk, dv, H); fp = CellParallelScan(dk, dv, H)
    # SAME inputs for both scans (an earlier version called inputs() twice and compared
    # gradients at different random points -- a test bug that manufactured an apparent
    # O(1) divergence and was misread as an implementation bug).
    d = inputs(); d2 = [t.clone() for t in d]
    gs = grads(sp, *d); gf = grads(fp, *d2)
    md = max((a-b).abs().max().item() for a,b in zip(gs,gf))
    check("fastscan gradient matches token-loop (same inputs)", md < 1e-8,
          f"max grad diff = {md:.3e}")


def test_credit_decay_is_the_pair_count_cliff():
    """RED-FLAG mechanism test (root cause of the 64-pair failure): the scan's credit
    assignment from a read back to a write decays as alpha^(t-s).

    Measured on this codebase: at the LEGACY init (alpha~0.5) the gradient at distance 128
    is ~1e-40 (underflows float64 -- dead credit: 64-pair MQAR can never bootstrap), while
    retention init (alpha~0.999) keeps it ~1e-2.  If this test fails, the recurrence's
    gating changed and long-range recall training will break again."""
    import torch.nn.functional as F
    from mulattice.sparse_scan import SparseLatticeScan

    def credit(T, a):
        torch.manual_seed(0)
        scan = SparseLatticeScan(8, 8, 1)
        addr = torch.zeros(1, T, 2, dtype=torch.float64)
        q2 = F.normalize(torch.randn(1, T, 1, 8, dtype=torch.float64), dim=-1)
        kap = F.normalize(torch.randn(1, T, 1, 8, dtype=torch.float64), dim=-1)
        v = torch.randn(1, T, 1, 8, dtype=torch.float64, requires_grad=True)
        alpha = torch.full((1, T, 1, 1), a, dtype=torch.float64)
        beta = torch.full((1, T, 1, 1), 0.9, dtype=torch.float64)
        out, _ = scan(addr, addr, q2, kap, v, alpha, beta, read_mode="all")
        out[0, T - 1].sum().backward()
        return v.grad[0, 0].abs().sum().item()

    dead = credit(128, 0.5)
    alive = credit(128, 0.999)
    check("legacy init kills 130-step credit (grad < 1e-30)", dead < 1e-30, f"{dead:.2e}")
    check("retention init keeps 130-step credit (grad > 1e-3)", alive > 1e-3, f"{alive:.2e}")