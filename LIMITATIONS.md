# Known Limitations

## Architecture

- **Density ceiling (~0.25 keys/cell).** The address scale must be set in advance so that the expected key count does not exceed this density. No adaptive mechanism exists during inference.
- **No collision recovery.** When multiple keys map to overlapping neighborhoods, the delta rule handles interference, but recall degrades above the density ceiling. Keys cannot be relocated during inference.
- **2D routing only.** Higher-dimensional routing would increase per-token cell access; the architecture uses exactly 2 dimensions and 9 cells.

## Evaluation scope

- **Position-indexed retrieval.** Content-addressed memory stores by content key, not position. Tasks requiring positional echo (Copy) score 0.003 without language pretraining, which teaches position as a side effect.
- **No language-modeling results.** All evaluations are associative-recall and real-text retrieval protocols. The mixer has not been tested as an attention replacement in LM pretraining.
- **Single-run reporting.** Capacity and forgetting results are from single training runs on single checkpoints. Variance across seeds was not measured for these protocols.
- **Vocabulary size.** Tested at 16,384 keys. The 131,072-pair benchmark (~262K keys) requires address scale ~385 (untested).
- **Overwrite density.** Write density (not just key density) governs collateral interference. At 0.43 writes/cell, untouched-key recall degrades to 0.93–0.96.

## Training

- **Address-scale changes.** Increasing the lattice scale during training on the hardest rung collapses the model. Scale changes must occur on easier rungs.
- **Training-memory bound.** The cell-parallel scan's autograd graph allocates ~rounds × n_cells cell copies. Training at large query counts requires proportionally more memory.
- **Vocabulary bootstrap.** Training a large vocabulary from random initialization does not escape chance-level loss. Warm-starting from a smaller vocabulary via donor-copy initialization is required.

## Integration

- **Bolt-on per-type scores.** The soft-prompt bridge to a frozen Qwen3.5-4B scores 1.000 on fresh facts whose answers come from a fixed value set (color, food, number, time) and 0.000 on fresh random strings (passwords, invented city names). Bindings are fresh at evaluation; the answer strings of fixed-set types necessarily repeat between training and evaluation.
- **Bridge trained once, single run.** Bolt-on results are from one training run on one checkpoint (selected by held-out fresh-fact exact-match); variance across training seeds was not measured.
- **Soft-prompt scale is small.** The trained bridge's gate settled at 0.003: soft tokens enter the frozen model at ~0.3% of embedding scale yet still flip the answer. Whether larger-scale injection improves open-vocabulary steering is untested.
