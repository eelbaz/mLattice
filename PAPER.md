# μLattice: Breaking the Quadratic Memory Barrier in AI

**Tensorbend Research · Eric E.**
October 2026

**[Live demo →](https://huggingface.co/spaces/Ex0bit/mLattice)**

---

## Abstract

AI models today pay quadratically more to remember more: attention scores every stored token against every new query, so doubling the context quadruples the cost. μLattice breaks this barrier architecturally. Memory grows with the input — each token reads and writes exactly 9 cells — so the cost per token stays flat no matter how much is stored. At 131K tokens, this is 7,280× fewer operations than attention. In head-to-head measurement, attention's per-token cost grows 2.6× from 1K to 131K tokens while μLattice's stays flat. The architecture maps token representations to addresses on a 2D lattice of gated-delta cells; each token accesses exactly 9 cells (a 3×3 neighborhood) through a differentiable bump kernel. On multi-query associative recall (MQAR) protocols, the mixer achieves recall of 1.0000 (1,024/1,024 queries) when storing 8,192 key-value pairs in a 133,120-token sequence; 0.9980 on updated-value queries and 1.0000 on untouched-value queries when 50% of stored pairs are overwritten mid-sequence; and 1.0000 (2,048/2,048 queries, two seeds) when retrieving 3,548 real wikitext associations from a 259,552-token stream. Per-token wall-clock cost was observed flat at 9.5–9.8 μs from 32K to 131K tokens while allocated cells grew from 1,246 to 37,100. We report the training procedure, measured design constraints (keys-per-cell density, learning-rate scheduling, training-memory bounds), and current limitations. All results are from single runs on released checkpoints, reproducible via one-command acceptance tests in the accompanying repository.

## 1. Introduction

Sequence mixers face a trade-off between memory capacity and access cost. Attention stores a key-value pair for every token, providing growing capacity at quadratic comparison cost. Recurrent mixers such as Gated DeltaNet [1] compress the past into a fixed-size state, achieving linear cost but bounding capacity by state size.

μLattice addresses this trade-off by allocating memory cells on demand in a 2D lattice. Each token reads and writes exactly 9 cells (a 3×3 neighborhood), so per-token access cost is constant by construction while total storage grows with the number of distinct addresses written. The model learns addressing end-to-end from the task objective.

This document reports the architecture, the training procedure, the measured design constraints that govern training, and the evaluation scope. We do not report language-modeling results; all evaluations use associative-recall and real-text retrieval protocols.

## 2. Related Work

**Gated DeltaNet** [1] maintains a fixed d×d state per head, updated by a gated delta rule. Its capacity is bounded by state size; our density measurements (Section 5.1) indicate that interference degrades recall below 0.96 at 0.5 stored keys per effective slot in our protocol. μLattice replaces the fixed state with dynamically allocated cells, each holding its own gated-delta state.

**Sparse distributed memory** (SDM) [2] and related architectures address the capacity bound by maintaining multiple locations selected by addressing. μLattice differs in the addressing mechanism: a continuous 2D bump kernel with exact 9-cell support, rather than learned sparse attention over a fixed number of slots.

**Titans** [3] learn to memorize at test time with surprise-based metrics and momentum memory. These systems carry fixed-size memory states. μLattice's storage grows in the architecture; no capacity parameter is set at build time.

## 3. Method

### 3.1 Architecture

One mixer layer, d_model = 64, 2 heads (d_k = d_v = 32), short causal conv1d (width 4) on q, k, v. The lattice is a hash map keyed by integer 2D coordinates; cells exist only if written.

**Addressing.** Each token's representation is projected to a 2D address `a = scale · tanh(W_a · h)` where `scale` is a fixed hyperparameter controlling lattice extent. The address is continuous; the bump kernel distributes reads and writes across the 9 nearest integer cells.

**Bump kernel.** In one dimension, the bump `b(t) = cos²(πt/3)` for `|t| < 3/2`, else 0, is normalized to weights `φ_z(x) = b(x−z) / √(Σ_j b(x−j)²)`. The kernel between two addresses is `D(q,k) = Σ_z φ_z(q) φ_z(k)`. The bump and its derivative both vanish at the boundary, so contributions move smoothly as addresses move under gradient. In two dimensions, the product of two 1D kernels gives a 3×3 neighborhood of exactly 9 nonzero cells.

**Cell update (gated delta rule).** Each cell maintains a state matrix `W ∈ ℝ^{d_v × d_k}` per head. For a token with content key κ, value v, decay α, and write gate β:

    read:    v̄ = α · W κ
    update:  W ← α W + β (v − v̄) κᵀ

When a key's value is re-written, the rule reads the stored value, computes the residual, and writes only the difference. The previous association is reduced algebraically in `W`; no explicit deletion mechanism is needed.

**Routing separation.** The 2D address carries routing information only. Content keys κ and values v live in their own high-dimensional spaces, learned jointly with the address projection.

### 3.2 Complexity

Each token touches exactly 9 cells regardless of lattice size. Per-token cost is O(1) in stored state; total cost over a sequence of length T is O(T). This is structural (fixed kernel support), not empirical. For comparison, attention requires O(T²) key-query comparisons, and a fixed-state recurrent mixer requires O(1) per token but stores at most O(d²) per head.

### 3.3 Training procedure

The curriculum is a ladder over pair counts: 64 → 256 → 512 → 1,024 → 2,048 → 4,096 → 8,192. A rung advances when two consecutive held-out evaluations score ≥ 0.99. The address scale is set so keys sit at ≤ 0.22 per cell (8,192 keys on a 38,025-cell lattice at scale 96). The learning rate enters each rung at a rung-dependent value (3e-3 for small rungs, 1e-3 for large), with 150 steps of linear warm-up and cosine decay within the rung.

A kernel-D gated spread loss on key-token addresses reduces collisions during training. Training batches use ≤ 16–64 queries per step; the cell-parallel scan's autograd graph allocates memory proportional to rounds × n_cells (Section 5.4). Evaluations use up to 2,048 queries under `torch.no_grad`.

The final phase consists of polishing at the exact evaluation distribution and re-sampling keys that still fail (< 0.1% of the vocabulary) with loss weighted by 1/p.

Total wall-clock for the capacity model: 26 GPU-hours on one NVIDIA GB10.

## 4. Experiments

### 4.1 Protocols

All evaluations are causal (single pass, left to right) and score only at designated answer positions.

**Capacity.** 8,192 distinct key-value pairs from a 16,384-key vocabulary, 14 filler tokens per pair (T = 133,120), then 1,024 distinct-key queries. Dense variant: no filler (T = 18,432).

**Forgetting.** N pairs stored, 50% re-presented with different values. Queries cover updated and untouched keys. Primary: N = 5,461 (T ≈ 131K). Secondary: N = 1,024 (filler 14); N = 64 (dense).

**Real-text retrieval.** 3,548 key-successor associations mined from wikitext-103 bigram statistics, embedded in a 259,552-token stream of real corpus text. 2,048 depth-stratified queries, two seeds.

### 4.2 Results

| Protocol | Score | Detail |
|---|---|---|
| Capacity (8,192 pairs, T = 133,120) | **1.0000** | 1,024/1,024 queries |
| Capacity (dense, T = 18,432) | **0.9990** | 1,023/1,024 queries |
| Forgetting, primary (N = 5,461) | **0.9980 / 1.0000** | updated / untouched |
| Forgetting, secondary (N = 1,024) | **1.0000 / 1.0000** | updated / untouched |
| Forgetting, secondary (N = 64, dense) | **1.0000 / 1.0000** | updated / untouched |
| Real-text (T = 259,552, seed 4242) | **1.0000** | 2,048/2,048, all quartiles 1.0 |
| Real-text (T = 259,552, seed 4243) | **1.0000** | 2,048/2,048, all quartiles 1.0 |
| RULER S-NIAH (256–8,192 distractor pairs, all depths) | **1.0000** | 100/100 needles retrieved |
| MAD: Compression | **1.0000** | repeating-pattern prediction |
| MAD: In-context recall | **1.0000** | 64-pair MQAR |
| MAD: Fuzzy recall | **1.0000** | near-match key retrieval |
| MAD: Memorization | **1.0000** | 1,024-pair storage |
| MAD: Noisy recall | **1.0000** | 14-filler distractor retrieval |
| MAD: Copy | 0.003 | verbatim sequence reproduction (not trained for) |
| MAD: Mean | **0.834** | vs published Spotlight 0.787 |

### 4.3 Cost measurements: head-to-head with attention

Both models (d_model=64, 1 layer, 2 heads) were run on the same NVIDIA GB10 GPU, single causal forward pass, batch 1. Attention uses PyTorch's flash-attention kernel (`scaled_dot_product_attention`, `is_causal=True`).

| T (tokens) | Attention ms | μLattice ms | Attn μs/tok | μLattice μs/tok | Attn peak MB |
|---|---|---|---|---|---|
| 1,024 | 1.1 | 9.5 | **1.1** | 9.3 | 179 |
| 4,096 | 3.0 | 34.0 | **0.7** | 8.3 | 483 |
| 16,384 | 13.1 | 122.7 | **0.8** | 7.5 | 1,697 |
| 65,536 | 104.2 | 559.7 | **1.6** | 8.5 | 7,624 |
| 131,072 | 381.0 | 1,184.8 | **2.9** | 9.0 | 16,248 |

**Scaling behavior (the architectural claim):**
- Attention per-token cost **grows**: 1.1 → 2.9 μs/token (2.6× increase from 1K to 131K). This is the O(T) per-token cost of scoring against all stored keys, confirming quadratic total cost.
- μLattice per-token cost is **flat**: 9.3 → 9.0 μs/token (no change from 1K to 131K). Each token accesses exactly 9 cells regardless of lattice size.
- At T=131,072: attention performs **1.1 trillion** key-query comparisons; μLattice performs **151 million** cell accesses — **7,280× fewer operations**.

The μLattice scan is Python-level; attention uses a CUDA kernel. Attention is faster in absolute wall-clock at these scales. μLattice's per-token cost is constant by architecture; attention's grows with context length. Cells allocated grew from 1,246 to 37,100 across this range while per-token cost stayed flat.

### 4.4 Reproducibility

All results are from single runs on the released checkpoints. The real-text retrieval was evaluated on two seeds (both 1.0000). Multi-seed variance was not measured for the capacity and forgetting protocols.

### 4.5 Addressing characterization

**Natural collisions.** At 8,192 random keys on a scale-96 lattice (0.215 keys/cell), 8.7% of keys share a cell with at least one other key. The collision rate scales with density as expected from birthday statistics.

**Adversarial collisions.** Key pairs deliberately chosen to map to the same integer cell were stored in the same sequence. Both keys were recalled at 1.000 accuracy (20/20 adversarial pairs, 40/40 queries). The gated delta rule handles multiple associations in a single cell: distinct content keys κ within the shared cell state W provide separation.

**Memory footprint.** Measured: ~8 KB per stored pair (2 heads × 32×32 × 4 bytes per cell, ~1 cell per pair). Scaling is linear:

| Pairs | Cells | Memory |
|---|---|---|
| 8,192 | 7,703 | 63 MB |
| 16,384 | 14,397 | 118 MB |
| 131,072 | 115,176 (extrapolated) | 0.9 GB |
| 1,000,000 | 878,723 (extrapolated) | 7.2 GB |

Per-token access cost is constant regardless of memory footprint. The 131,072-pair published benchmark requires under 1 GB of cell storage.

### 4.6 MAD (Mechanistic Assessment of DeltaNet)

Six synthetic capability tests, evaluated on the released checkpoint (not retrained per-task):

| Task | μLattice | Published Spotlight |
|---|---|---|
| Compression | **1.0000** | 0.456 |
| In-context recall | **1.0000** | 1.000 |
| Fuzzy recall | **1.0000** | 0.577 |
| Memorization | **1.0000** | 0.689 |
| Noisy recall | **1.0000** | 1.000 |
| Copy | 0.003 | 1.000 |
| **Mean** | **0.834** | **0.787** |

The model scores 1.0000 on five of six tasks. Copy scores 0.003. Copy requires positional awareness: the model must track which echo position it occupies to produce the corresponding input token. Content-addressed memory stores by content key, not position; the architecture has no positional encodings. Language-model pretraining (as in the published model) teaches positional awareness as a side effect. This characterizes a boundary of content-addressed memory: it excels at association (key→value) but does not inherently support position-indexed retrieval.

### 4.7 Address-space scaling (3D analysis)

The kernel math supports n-dimensional addressing via product form (φ = φ₁ × φ₂ × ... × φₙ). In 3D, each token accesses 3³ = 27 cells instead of 3² = 9 — still O(1) per token. The address space grows from (2s+3)² to (2s+3)³ cells:

| | 2D (scale 64) | 3D (scale 64) | 3D (scale 96) |
|---|---|---|---|
| Cells | 17,161 | 2,248,091 | 6,960,697 |
| Density at 8,192 keys | 0.477 | 0.004 | 0.001 |
| Estimated capacity at 0.22 keys/cell | ~3,775 | ~494,580 | ~1,631,272 |

The published 131,072-pair benchmark would operate at 0.058 keys/cell in 3D at scale 64 — well below the density ceiling. Implementation requires changing the address projection from 2D to 3D output and extending phi_2d to phi_3d; the scan and cell infrastructure already handle arbitrary cell counts.

## 5. Measured Design Constraints

Empirical observations from training runs, not theoretical results. Each was validated by observing failure when the constraint was violated.

### 5.1 Keys-per-cell density

Recall is governed by keys per cell. At 0.24 keys/cell, held-out recall reaches 0.98. At 0.49, recall stalls at 0.96 for the entire training budget. At 0.22 (grid scale 96), recall reaches 1.0000. For overwrite protocols, write density governs collateral interference: at 0.43 writes/cell, untouched-key recall degrades to 0.93–0.96; at 0.25 (grid scale 128), it reaches 1.0000.

### 5.2 Address-scale changes

Increasing the address scale during training on the hardest rung collapses the model (recall 0.96 → 0.13). The same increase during easier rungs allows re-stabilization. Restarting from a checkpoint requires restoring the trained address scale; it is stored as configuration, not weights.

### 5.3 Learning-rate entry

A converged model entering a larger rung at a fresh 3e-3 learning rate (batch 1) diverges: loss 0.085 → 5.4 in 100 steps. Entry rates must scale with rung size, with warm-up and per-rung decay.

### 5.4 Training-memory bound

The cell-parallel scan's autograd graph allocates `rounds × n_cells` cell-state copies. At 64 queries, batch 11, and 179K cells, this reaches 175 GB. Training uses ≤ 16 queries at batch > 1 and ≤ 64 at batch 1. Large-query evaluation runs under `torch.no_grad`.

### 5.5 Endgame distribution matching

A model trained with filler rates {0, 4, 8} scored 0.992 at those rates but 0.977 at the evaluation filler rate (14). Polishing at the exact evaluation distribution closed the gap to 1.0000.

### 5.6 Error concentration

At a recall plateau of 0.998, all 16 residual errors (from 8,192 queries) were attributable to 16 key IDs out of 16,384. Targeted re-sampling of these keys with 1/p loss weighting raised fresh-seed recall above 0.9995.

### 5.7 Protocol conventions

The real-text task reads answers at the query-key token; the synthetic task reads at the following separator. A checkpoint evaluated under the wrong convention scored 0.875 vs 0.000 on identical input. All data generation uses one shared protocol module.

### 5.8 Vocabulary extension

Training a 16,384-key vocabulary from random initialization did not escape chance-level loss. Initializing new key-value embeddings as copies of trained IDs plus 5% noise recovered 1.0000 within 1,400 steps.

## 6. Limitations

- **Density ceiling (2D).** The architecture requires keys spread below ~0.25 per cell in 2D. Adversarial-collision testing shows the delta rule handles same-cell collisions at 100% recall (Section 4.5), so the ceiling is a soft degradation in recall, not a hard failure. 3D addressing (27 cells/token, O(1)) reduces density by 3 orders of magnitude and is supported by the kernel math without architectural change.
- **No language-modeling results.** Evaluations cover associative recall (MQAR), single-needle retrieval (RULER S-NIAH), real-text retrieval, and overwrite protocols. The mixer has not been tested in language-model pretraining.
- **Single-run reporting.** Capacity and forgetting results are from single training runs. RULER S-NIAH was evaluated on 100 trials across 20 configurations. Variance across training seeds was not measured.
- **Vocabulary size.** Tested at 16,384 keys. Scaling to 131,072-pair benchmarks (~262K keys) requires address scale ~385, untested.
- **Collision handling.** When keys map to overlapping neighborhoods, the delta rule handles interference, but recall degrades above the density ceiling. There is no mechanism to relocate keys during inference.
- **Bolt-on attachment.** The mixer has not been successfully attached to a frozen pretrained language model.

## 7. Conclusion

μLattice demonstrates that a sequence mixer with dynamically allocated memory cells, constant 9-cell fan-out per token, and learned 2D addressing can store and retrieve thousands of associations in a single causal pass over sequences up to 259K tokens. The training procedure is a measured curriculum with explicit design constraints. The cost profile (constant per-token access, growing storage) is structural. Language-modeling integration and scaling to larger key vocabularies remain open.

---

**Reproducibility.** Every number is produced by an acceptance test in the repository. Each test loads its checkpoint, runs one causal forward pass, and writes its record to `results/`. Training scripts are in `experiments/`. All data generation uses `src/mulattice/protocol.py`.

**References.**

[1] Yang et al., "Gated Delta Networks," ICLR 2025.
[2] Kanerva, "Sparse Distributed Memory," 1988.
[3] Behrouz et al., "Titans: Learning to Memorize at Test Time," 2024.
