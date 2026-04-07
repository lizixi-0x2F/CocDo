"""CausalFFNN and NodeProjector — pairwise causal weight discovery and scalar projection."""
from __future__ import annotations
import torch
import torch.nn as nn


class CausalFFNN(nn.Module):
    """
    Input : E ∈ R^{N × D}  (token embedding matrix)
    Output: A ∈ R^{N × N}  causal weight matrix, A[i,j] ∈ (0,1)

    Each entry A[i,j] is an independent sigmoid gate: the probability that
    node i is a parent of node j. Diagonal is forced to 0 (no self-loops).

    Sigmoid replaces column-softmax: edges are not in competition, so weak
    edges can go to 0 independently, giving natural sparsity without L1.
    """

    def __init__(self, d_embed: int, hidden: int = 256):
        super().__init__()
        self._score_dim = hidden
        self.token_enc = nn.Sequential(
            nn.Linear(d_embed, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.pair_score = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # Negative bias initialisation: edges start near 0 (sparse prior)
        last = self.pair_score[-1]
        assert isinstance(last, nn.Linear)
        nn.init.normal_(last.weight, std=0.1)
        nn.init.constant_(last.bias, 0.0)    # sigmoid(0) = 0.5，无稀疏先验

    def forward(self, E: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """E: (B, N, D) or (N, D) → (A, logits).

        When E is (B, N, D), token representations are aggregated across the
        batch via mean+max pooling before pairwise scoring, so A reflects the
        full sample distribution rather than a single mean embedding.

        A[i,j] = sigmoid(logits[i,j]), diagonal zeroed (no self-loops).
        """
        if E.dim() == 2:
            E = E.unsqueeze(0)          # (1, N, D)
        B, N, _ = E.shape
        H = self.token_enc(E.reshape(B * N, -1)).reshape(B, N, -1)  # (B, N, hidden)
        # Aggregate across batch: mean + max → (N, 2*hidden)
        H_mean = H.mean(dim=0)          # (N, hidden)
        H_max  = H.max(dim=0).values    # (N, hidden)
        H_agg  = H_mean + H_max         # (N, hidden)  sum keeps dim for pair_score

        Hi = H_agg.unsqueeze(1).expand(N, N, -1)   # (N, N, hidden)
        Hj = H_agg.unsqueeze(0).expand(N, N, -1)   # (N, N, hidden)
        logits = self.pair_score(
            torch.cat([Hi, Hj], dim=-1)
        ).squeeze(-1)                               # (N, N)

        logits = logits / (self._score_dim ** 0.5)

        diag_mask = torch.eye(N, device=E.device, dtype=torch.bool)
        logits = logits.masked_fill(diag_mask, float("-inf"))
        A = torch.sigmoid(logits)
        A = A.masked_fill(diag_mask, 0.0)
        return A, logits


class NodeProjector(nn.Module):
    """
    Projects each node's D-dimensional embedding to a scalar observation value.

    Bridges the embedding space and observation space:
        y_j = w^T tanh(E_j) + b  ∈ R

    This grounds the structural equation E_j = A[:,j]@E + U_j in observation
    space, making infer_effect() and counterfactual() return interpretable
    scalars rather than raw embedding norms.

    Parameters
    ----------
    d_embed : int
        Dimension of input node embeddings (must match CausalFFNN.d_embed).
    """

    def __init__(self, d_embed: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.Tanh(),
            nn.Linear(d_embed, 1),
        )

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        """E: (N, D) → y: (N,) scalar observation per node."""
        return self.proj(E).squeeze(-1)

