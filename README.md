# μLattice

**Breaking the quadratic memory barrier in AI.** Today's models pay quadratically more to remember more. μLattice grows memory linearly: each token reads and writes exactly 9 cells on a 2D lattice, so per-token cost is flat by architecture. At 131K tokens, this is 7,280× fewer operations than attention (measured head-to-head on the same GPU).

**Tensorbend Research · Eric E. · October 2026**

## Installation

```bash
# requirements
python >= 3.10
torch >= 2.4
CUDA GPU (recommended for acceptance tests; CPU works for unit tests)

pip install -e .
```

**[Live demo on Hugging Face →](https://huggingface.co/spaces/Ex0bit/mLattice)**

## Quickstart (no training required)

```bash
# unit tests: kernel invariants, scan equivalence (runs on CPU, ~30s)
python3 tests/test_kernel.py

# acceptance tests (require CUDA GPU + checkpoints in checkpoints/)
python3 tests/test_capacity.py cuda     # ~5s
python3 tests/test_forgetting.py cuda   # ~7s
python3 tests/test_realtext.py cuda     # ~30s (requires data/wikitext103_slice.txt)
```

## Results

All scores from single runs on released checkpoints.

| Protocol | Score | Test |
|---|---|---|
| Capacity: 8,192 pairs, T=133,120, 1,024 queries | **1.0000** (dense: 0.9990) | `tests/test_capacity.py` |
| Forgetting: 50% overwritten, T≈131K | **0.9980 / 1.0000** (updated/untouched) | `tests/test_forgetting.py` |
| Real-text: 3,548 wiki pairs, T=259,552, 2×2,048 queries | **1.0000**, both seeds, all depths | `tests/test_realtext.py` |
| RULER S-NIAH: single needle in 256–8,192 distractor pairs | **1.0000** (100/100, all depths) | `experiments/eval_ruler.py` |
| MAD: six-task mechanistic suite | **0.834 mean** (5/6 at 1.0000) | `experiments/eval_mad.py` |
| Bolt-on: frozen Qwen3.5-4B + μLattice soft tokens | **0.645 vs 0.025 floor** (400 fresh-fact trials, 2 seeds) | `experiments/eval_bolt_on.py` |

### Cost

Per-token wall-clock measured flat at 9.5–9.8 μs from 32K to 131K tokens while allocated cells grew from 1,246 to 37,100. Constant 9-cell fan-out per token is structural (fixed kernel support), not empirical.

## Bolt-on: memory that steers a frozen LLM

A trainable 10.8M-parameter bridge (`src/mulattice/bridge.py`) reads the frozen lattice's 64-d cell readout at a query and emits 8 soft prompt tokens prepended to a frozen Qwen3.5-4B — the only trainable component; the LM and the lattice stay bit-identical (verified by hashing before/after training).

- Documents are ingested as string-hashed key→value pairs under the lattice's trained protocol; retrieval on evaluation facts measured 1.0000.
- Fresh-fact evaluation (entities and bindings never seen in training, 200 cases × 2 seeds): exact-match **0.645** with soft tokens vs **0.025** for the frozen model alone, **0.015** with shuffled memory (content-addressing control), and **1.000** for RAG with the document pasted into the context.
- Score by fact type: color, food, number and time at **1.000**; password and city types at **0.000**.

```bash
python3 experiments/train_bridge.py      # Stage A + Stage B (~90 min on a GB10)
python3 experiments/eval_bolt_on.py      # floor / RAG ceiling / test / control
python3 tests/test_bridge.py             # bridge unit tests (CPU, ~10s)
```

## Checkpoints

| File | Size | Description |
|---|---|---|
| `checkpoints/capacity8k.pt` | 17 MB | capacity model (8,192 pairs) |
| `checkpoints/forgetting.pt` | 17 MB | overwrite model |
| `checkpoints/realtext.pt` | 4.4 MB | real-text retrieval model |
| `checkpoints/bridge.pt` | 43 MB | lattice→Qwen3.5 soft-prompt bridge (bolt-on) |
| `checkpoints/mqar_base.pt` | 1.2 MB | warm-start ancestor |
| `checkpoints/realtext_base.pt` | 4.4 MB | real-text warm-start |

Approximate GPU-hours: capacity ~26h, forgetting ~12h, real-text ~6h (single NVIDIA GB10).

## Training

```bash
python3 experiments/train_capacity.py       # curriculum to 8,192 pairs
python3 experiments/train_forgetting.py     # overwrite curriculum
python3 experiments/train_realtext.py       # real-text retrieval
```

One protocol module (`src/mulattice/protocol.py`) generates all training and evaluation data.

## Demo

```bash
python3 experiments/demo_server.py          # open http://localhost:8772
```

Interactive: paste your own text, define named facts, mark overwrites, query the model. Shows the learned address map, measured cost curve, and cell-state inspection.

## Full write-up

See [`PAPER.md`](PAPER.md) for the complete research note (architecture, measured design constraints, limitations).

## Cite

```bibtex
@misc{mulattice2026,
  title  = {μLattice: A Dynamically Allocated Memory Lattice for Sequence Models},
  author = {Eric E. and Tensorbend Research},
  year   = {2026},
  url    = {github.com/eelbaz/mLattice}
}
```
