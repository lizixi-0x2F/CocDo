"""CocNN — Unified causal inference model: FFN causal discovery + COC type-theoretic intervention."""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

from .causal_ffnn import CausalFFNN
from .scm import NeuralSCM


class CocNN(nn.Module):
    """
    Unified causal neural network.

    Pipeline
    --------
    E (N×D embeddings)
      → SeqCausalFFNN  → A (N×N causal weight matrix, upper-triangular DAG)
      → NeuralSCM      → COC-encoded structural causal model
      → do / counterfactual / infer_effect

    Parameters
    ----------
    d_embed : int
        Dimension of input token embeddings.
    hidden : int
        Hidden size for the causal FFN.
    """

    def __init__(self, d_embed: int, hidden: int = 256):
        super().__init__()
        self.ffnn = CausalFFNN(d_embed, hidden)
        self.scm: Optional[NeuralSCM] = None
        self._var_names: Optional[list[str]] = None

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        """Return the causal weight matrix A ∈ [0,1]^{N×N}."""
        return self.ffnn(E)

    def build(self, E: torch.Tensor, var_names: list[str]) -> NeuralSCM:
        """
        Build the full causal model from embeddings.

        1. FFN infers causal weight matrix A (column-softmax, upper-triangular)
        2. Compute exogenous residuals U = E - A^T @ E
        3. Construct NeuralSCM with COC type encoding
        4. Add soft causal edges with softmax weights as causal strength

        Each A[i,j] > 0 in the upper triangle becomes a soft causal edge
        from var_names[i] → var_names[j] with weight A[i,j]. The softmax
        weights are encoded as COC Const terms carrying continuous causal
        strength, enabling differentiable causal reasoning.

        Returns the constructed NeuralSCM (also stored as self.scm).
        """
        with torch.no_grad():
            A_t = self.ffnn(E)
        A = A_t.detach().cpu().numpy()
        E_np = E.detach().cpu().numpy()
        U = E_np - A.T @ E_np

        self._var_names = list(var_names)
        self.scm = NeuralSCM(var_names, A, E_np, U)

        # Soft causal edges: all non-diagonal positions with non-zero weight
        N = len(var_names)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                w = float(A[i, j])
                if w > 0:
                    self.scm.add_causal_edge(var_names[i], var_names[j], weight=w)

        return self.scm

    def do(self, var: str, value: float) -> NeuralSCM:
        """COC causal intervention do(var = value)."""
        if self.scm is None:
            raise RuntimeError("Call build() before do()")
        return self.scm.do(var, value)

    def counterfactual(self, interventions: dict[str, float], target: str) -> float:
        """Pearl three-step counterfactual inference."""
        if self.scm is None:
            raise RuntimeError("Call build() before counterfactual()")
        return self.scm.counterfactual(interventions, target)

    def infer_effect(self, target: str) -> float:
        """Structural equation causal effect estimation."""
        if self.scm is None:
            raise RuntimeError("Call build() before infer_effect()")
        return self.scm.infer_effect(target)

    def causal_graph(self) -> dict[str, list[str]]:
        """Return the discovered causal graph as {child: [parents]}."""
        if self.scm is None:
            raise RuntimeError("Call build() before causal_graph()")
        return {name: list(node.parents) for name, node in self.scm.vars.items()}
