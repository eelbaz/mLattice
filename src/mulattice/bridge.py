"""Bridge: projects frozen μLattice cell readouts into a frozen LM's soft prompt space.

The bridge is the ONLY trainable component of the bolt-on.  It reads the
64-d cell readout the μLattice produces at a query's answer position and
emits ``n_soft_tokens`` vectors in the language model's embedding space,
prepended to the query exactly like ordinary input embeddings.  A scalar
gate multiplies the output and starts at zero, so an untrained bridge is a
no-op for the frozen model and training opens the channel gradually.
"""
from __future__ import annotations

import torch
import torch.nn as nn

DEFAULT_LATTICE_DIM = 64      # μLattice d_model (capacity8k checkpoint)
DEFAULT_LM_HIDDEN = 2560      # Qwen3.5-4B hidden_size
DEFAULT_N_SOFT_TOKENS = 8     # soft prompt tokens prepended to the query
DEFAULT_HIDDEN = 512          # bridge MLP hidden width
POSITION_INIT_STD = 0.02      # std of the learnable per-slot position offsets
GATE_INIT_LOGIT = -6.0        # sigmoid(logit)=2.5e-3: closed at init, but the
                              # multiplicative gate still passes upstream
                              # gradients from step 0 (a hard zero gate blocks
                              # every gradient except its own)


class LatticeBridge(nn.Module):
    """Converts a frozen μLattice readout (B, lattice_dim) into K soft prompt
    tokens (B, K, lm_hidden) for a frozen causal language model."""

    def __init__(self, lattice_dim: int = DEFAULT_LATTICE_DIM,
                 lm_hidden: int = DEFAULT_LM_HIDDEN,
                 n_soft_tokens: int = DEFAULT_N_SOFT_TOKENS,
                 hidden: int = DEFAULT_HIDDEN):
        super().__init__()
        self.n_soft = n_soft_tokens
        self.lm_hidden = lm_hidden
        self.read_proj = nn.Sequential(
            nn.Linear(lattice_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.token_proj = nn.Linear(hidden, n_soft_tokens * lm_hidden)
        self.token_positions = nn.Parameter(
            torch.randn(n_soft_tokens, lm_hidden) * POSITION_INIT_STD)
        self.gate = nn.Parameter(torch.full((1,), GATE_INIT_LOGIT))

    def forward(self, lattice_readout: torch.Tensor) -> torch.Tensor:
        """Map lattice cell readouts to gated soft prompt tokens.

        lattice_readout: (B, lattice_dim) — the W·κ readout vector the frozen
            lattice produced at the query's answer position.
        returns: (B, n_soft_tokens, lm_hidden) — embeddings prepended to the
            frozen model's input sequence.
        """
        h = self.read_proj(lattice_readout)
        tokens = self.token_proj(h).view(-1, self.n_soft, self.lm_hidden)
        tokens = tokens + self.token_positions.unsqueeze(0)
        return torch.sigmoid(self.gate) * tokens
