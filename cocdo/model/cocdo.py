"""CocNN — Unified causal inference model: FFN causal discovery + COC type-theoretic intervention."""
from __future__ import annotations
import logging
from typing import Optional

import torch
import torch.nn as nn

from .causal_ffnn import CausalFFNN, NodeProjector
from .scm import NeuralSCM

logger = logging.getLogger(__name__)


class CocDo(nn.Module):
    """
    Unified causal neural network with batch training support.

    Architecture
    ------------
    Node identity is represented by a learnable position embedding P ∈ R^{N×D}.
    Per-sample observation values are injected via a learnable value scale v ∈ R^D:

        E_b = P + y_b[:, None] * v          (B, N, D)

    This gives each sample a unique embedding that carries both node identity
    and the observed value, letting CausalFFNN learn graph structure from
    cross-sample covariance.

    CausalFFNN operates on the mean embedding Ē = E_b.mean(0) ∈ R^{N×D} to
    produce a single shared causal weight matrix A ∈ R^{N×N} (sigmoid per edge).
    The structural equation per sample is:

        ŷ_b = projector(A^T @ E_b)

    Training objective
    ------------------
    L = L_recon = mean over batch of MSE(projector(E_b), y_b)

    Sparsity comes from sigmoid initialisation (bias=-2, starts near 0) and
    recon pressure: edges that don't help prediction stay near 0.

    Edge selection at build() time: A[i,j] > 0.5.

    Parameters
    ----------
    d_embed : Dimension of node embeddings.
    hidden  : Hidden size for CausalFFNN (also attention scale ÷√hidden).
    """

    def __init__(
        self,
        d_embed: int,
        hidden: int = 256,
    ):
        super().__init__()
        self.d_embed      = d_embed
        self.ffnn         = CausalFFNN(d_embed, hidden)
        self.projector    = NodeProjector(d_embed)
        self.scm: Optional[NeuralSCM] = None
        self._var_names: Optional[list[str]] = None

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        """E: (N, D) → A: (N, N) column-softmax causal weight matrix."""
        A, _ = self.ffnn(E)
        return A

    def encode(
        self,
        y_batch: torch.Tensor,           # (B, N)
        pos_emb: torch.Tensor,           # (N, D)  learnable node identity
        val_scale: torch.Tensor,         # (D,)    learnable value direction
    ) -> torch.Tensor:
        """
        Encode a batch of observations into node embeddings.

        E_b = pos_emb + y_b[:, None] * val_scale    shape (B, N, D)
        """
        return pos_emb.unsqueeze(0) + y_batch.unsqueeze(-1) * val_scale  # (B, N, D)

    def train_step(
        self,
        y_batch: torch.Tensor,           # (B, N)  observed scalar values
        pos_emb: torch.Tensor,           # (N, D)  learnable node position embedding
        val_scale: torch.Tensor,         # (D,)    learnable value direction
    ) -> dict[str, torch.Tensor]:
        """
        Single batch training step.

        A is inferred from the mean embedding (node identity + mean signal),
        then the structural equation is evaluated per sample to compute recon loss.

        Parameters
        ----------
        y_batch   : (B, N) batch of joint observations
        pos_emb   : (N, D) learnable node identity embeddings
        val_scale : (D,)   learnable value encoding direction

        Returns
        -------
        dict with keys: loss, loss_recon
        """
        B, N = y_batch.shape

        # Per-sample embeddings: (B, N, D)
        E_batch = self.encode(y_batch, pos_emb, val_scale)

        # Shared graph from mean embedding (N, D)
        E_mean = E_batch.mean(dim=0)
        A, _ = self.ffnn(E_mean)                     # (N, N)

        # Structural equation per sample: ŷ_b = projector(A^T @ E_b)
        E_hat = torch.einsum("ij,bjd->bid", A.T, E_batch)   # (B, N, D)
        y_hat = self.projector(E_hat.reshape(B * N, -1)).reshape(B, N)  # (B, N)

        loss_recon = nn.functional.mse_loss(y_hat, y_batch)
        return {
            "loss":       loss_recon,
            "loss_recon": loss_recon,
        }

    def build(
        self,
        pos_emb: torch.Tensor,           # (N, D)
        var_names: list[str],
        y_mean: Optional[torch.Tensor] = None,   # (N,) optional: add mean signal
        val_scale: Optional[torch.Tensor] = None,
        topo_order: Optional[list[str]] = None,  # known topological order, roots first
    ) -> NeuralSCM:
        """
        Build the NeuralSCM from the learned position embeddings.

        Uses E = pos_emb (+ y_mean * val_scale if provided) as the reference
        embedding for structural equations.

        Edge selection: A[i,j] > 0.5.

        If topo_order is provided, Sort levels are assigned from it (position =
        level), and add_causal_edge will reject any edge that violates the order.
        Edges discovered by CausalFFNN that conflict with topo_order are silently
        skipped — the CoC type guard catches them and build() continues.
        """
        with torch.no_grad():
            if y_mean is not None and val_scale is not None:
                E = pos_emb + y_mean.unsqueeze(-1) * val_scale
            else:
                E = pos_emb
            A_t, _ = self.ffnn(E)
        A    = A_t.detach().cpu().numpy()
        E_np = E.detach().cpu().numpy()
        U    = E_np - A.T @ E_np

        self._var_names = list(var_names)
        self.scm = NeuralSCM(var_names, A, E_np, U, topo_order=topo_order)

        N = len(var_names)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                w = float(A[i, j])
                if w > 0.5:
                    try:
                        self.scm.add_causal_edge(var_names[i], var_names[j], weight=w)
                    except TypeError:
                        # Edge violates topo_order — discard it.
                        logger.debug(
                            "build(): skipping edge %s→%s (topo violation)",
                            var_names[i], var_names[j],
                        )

        return self.scm

    def project(self, E: torch.Tensor) -> torch.Tensor:
        """Project node embeddings to scalar observations: (N, D) → (N,)."""
        with torch.no_grad():
            return self.projector(E)

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