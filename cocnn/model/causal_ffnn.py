"""SeqCausalFFNN — pairwise FFN causal weight discovery."""
from __future__ import annotations
import torch
import torch.nn as nn


class CausalFFNN(nn.Module):
    """
    Input : E ∈ R^{N × D}  (token embedding matrix)
    Output: A ∈ [0,1]^{N × N}  causal weight matrix (no ordering constraint)

    Each entry A[i,j] = sigmoid(score(h_i, h_j)), representing the causal
    strength from variable i to variable j. No upper-triangular or softmax
    constraint — the graph structure is learned freely from data.
    """

    def __init__(self, d_embed: int, hidden: int = 256):
        super().__init__()
        self.token_enc = nn.Sequential(
            nn.Linear(d_embed, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.pair_score = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        """E: (N, D) → A: (N, N) where A[i,j] = sigmoid weight of edge i→j."""
        N, _ = E.shape
        H  = self.token_enc(E)                              # (N, hidden)
        Hi = H.unsqueeze(1).expand(N, N, -1)               # (N, N, hidden)
        Hj = H.unsqueeze(0).expand(N, N, -1)               # (N, N, hidden)
        logits = self.pair_score(
            torch.cat([Hi, Hj], dim=-1)
        ).squeeze(-1)                                       # (N, N)

        # Zero out diagonal (no self-loops), keep all other edges free
        diag_mask = torch.eye(N, device=E.device).bool()
        logits = logits.masked_fill(diag_mask, float("-inf"))
        # softplus: smooth, always positive, always has gradient (no dead neurons)
        return torch.nn.functional.softplus(logits)
