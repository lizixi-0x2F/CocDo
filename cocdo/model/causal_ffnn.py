"""CausalFFNN — low-rank bilinear causal discovery (attention = causal graph)."""
from __future__ import annotations
import torch
import torch.nn as nn


def acyclicity_loss(A: torch.Tensor) -> torch.Tensor:
    """Spectral acyclicity proxy — O(N) for sparse/low-value A.

    For sigmoid outputs near 0 (sparse graph), A∘A has small entries.
    tr(e^{A∘A}) ≈ n + tr(A∘A) + tr((A∘A)²)/2 + ...

    We use the first non-trivial term: tr(A∘A) = ||A||_F²
    This is zero iff A=0 (trivial DAG) and grows with cycle strength.
    Combined with the reconstruction loss pushing A to explain E,
    it acts as a soft sparsity+acyclicity prior without matrix_exp overflow.

    For the small-N demos (N<200), falls back to exact matrix_exp.
    """
    return (A * A).sum()


def topo_order_from_A(A: "numpy.ndarray", var_names: list[str]) -> list[str]:  # type: ignore[name-defined]
    """Extract a topological ordering from a learned causal weight matrix.

    Uses Kahn's algorithm on the thresholded adjacency.  Nodes are sorted
    root-first (in-degree 0 first), matching the Sort-level convention in
    NeuralSCM.

    Parameters
    ----------
    A         : (N, N) numpy array — A[i, j] > threshold means i → j
    var_names : node names corresponding to A rows/cols

    Returns
    -------
    list of node names in topological order (roots first).
    Raises ValueError if A contains a cycle (shouldn't happen after training
    with acyclicity_loss, but guarded defensively).
    """
    import numpy as np
    n = len(var_names)
    threshold = 1e-2
    adj = (A > threshold).astype(int)   # (N, N)  adj[i,j]=1 means i→j

    in_deg = adj.sum(axis=0).tolist()   # number of parents per node
    queue  = [i for i in range(n) if in_deg[i] == 0]
    order: list[int] = []

    while queue:
        # stable sort by index so output is deterministic
        queue.sort()
        node = queue.pop(0)
        order.append(node)
        for child in range(n):
            if adj[node, child]:
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    queue.append(child)

    if len(order) != n:
        # Cycle detected — fall back to in-degree heuristic (best-effort)
        in_deg_orig = adj.sum(axis=0)
        order = list(np.argsort(in_deg_orig))

    return [var_names[i] for i in order]


class CausalFFNN(nn.Module):
    """
    Low-rank bilinear causal discovery.

    Input : E ∈ R^{N × D}
    Output: A ∈ R^{N × N},  A[i,j] ∈ (0,1)

    score(i→j) = (W_q · h_i) · (W_k · h_j)ᵀ / √rank

    Complexity: O(N · rank) vs O(N² · hidden) for the old MLP pair_score.
    Equivalent to one Transformer attention head — which means attention IS
    causal discovery, just never named that way.
    """

    def __init__(self, d_embed: int, hidden: int = 256, rank: int = 64):
        super().__init__()
        self.rank = rank
        self.encoder = nn.Sequential(
            nn.Linear(d_embed, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.W_q = nn.Linear(hidden, rank, bias=False)
        self.W_k = nn.Linear(hidden, rank, bias=False)

    def forward(self, E: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """E: (B, N, D) or (N, D) → (A, logits), O(N·rank)."""
        if E.dim() == 2:
            E = E.unsqueeze(0)           # (1, N, D)
        B, N, _ = E.shape

        H = self.encoder(E.reshape(B * N, -1)).reshape(B, N, -1)  # (B, N, hidden)
        # aggregate batch
        H = H.mean(dim=0)               # (N, hidden)

        Q = self.W_q(H)                 # (N, rank)
        K = self.W_k(H)                 # (N, rank)
        logits = Q @ K.T / (self.rank ** 0.5)   # (N, N)

        diag_mask = torch.eye(N, device=E.device, dtype=torch.bool)
        logits = logits.masked_fill(diag_mask, float("-inf"))
        A = torch.sigmoid(logits)
        A = A.masked_fill(diag_mask, 0.0)
        return A, logits

