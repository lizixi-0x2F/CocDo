"""CausalNetwork — end-to-end differentiable causal reasoning network.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │ Input: observations X ∈ ℝ^{B×N×D}                           │
    │         interventions dict[str, float]                      │
    │         targets dict[str, float]                            │
    ├─────────────────────────────────────────────────────────────┤
    │ 1. Causal Discovery Layer (CausalFFNN)                      │
    │    A, _ = causal_ffnn(X)       # (N, N) causal weight matrix│
    ├─────────────────────────────────────────────────────────────┤
    │ 2. Structural Causal Model (NeuralSCM)                      │
    │    scm = NeuralSCM.from_embeddings(var_names, A, X)         │
    ├─────────────────────────────────────────────────────────────┤
    │ 3a. Intervention Effect Prediction                          │
    │    state, E_next = scm.step(interventions)                  │
    │ 3b. Counterfactual Inference                                │
    │    cf_value = scm.counterfactual(interventions, target)     │
    │ 3c. Causal Planning                                         │
    │    planner = CausalPlanner(scm)                             │
    │    result = planner.plan(E_init, target, interv_nodes)      │
    └─────────────────────────────────────────────────────────────┘

This class provides a unified interface for:
    - Learning causal structure from observational data
    - Predicting intervention effects
    - Counterfactual reasoning
    - Gradient-based causal planning

All operations are differentiable w.r.t. the causal weights.
"""

from __future__ import annotations
from typing import Optional, Union, Dict, List, Tuple
import logging

import numpy as np
import torch
import torch.nn as nn

from .causal_ffnn import CausalFFNN, acyclicity_loss, topo_order_from_A
from .scm import NeuralSCM
from .planner import CausalPlanner

logger = logging.getLogger(__name__)


class CausalNetwork(nn.Module):
    """End-to-end differentiable causal reasoning network.

    This network learns causal structure from data and supports:
    - Intervention effect prediction
    - Counterfactual inference
    - Gradient-based causal planning

    Parameters
    ----------
    d_embed : int
        Dimensionality of node embeddings
    n_vars : int
        Number of variables/nodes
    var_names : list[str]
        Names of variables (length n_vars)
    hidden : int, default=256
        Hidden dimension for CausalFFNN
    rank : int, default=64
        Low-rank dimension for bilinear attention in CausalFFNN
    enforce_acyclic : bool, default=True
        Whether to enforce acyclicity during training
    """

    def __init__(
        self,
        d_embed: int,
        n_vars: int,
        var_names: list[str],
        hidden: int = 256,
        rank: int = 64,
        enforce_acyclic: bool = True,
    ):
        super().__init__()
        self.d_embed = d_embed
        self.n_vars = n_vars
        self.var_names = var_names
        self.enforce_acyclic = enforce_acyclic

        # Causal discovery module
        self.causal_ffnn = CausalFFNN(d_embed=d_embed, hidden=hidden, rank=rank)

        # Placeholders for SCM and planner (built after training)
        self._scm: Optional[NeuralSCM] = None
        self._planner: Optional[CausalPlanner] = None
        self._A: Optional[torch.Tensor] = None
        self._E: Optional[torch.Tensor] = None
        self._U: Optional[torch.Tensor] = None

        # Augmented Lagrangian parameters for acyclicity constraint
        self.register_buffer("_rho", torch.tensor(1.0))
        self.register_buffer("_lam", torch.tensor(0.0))
        self._h_prev = float("inf")

    def forward(
        self,
        X: torch.Tensor,
        mode: str = "train",
        interventions: Optional[Dict[str, float]] = None,
        target: Optional[Union[str, Dict[str, float]]] = None,
        interv_nodes: Optional[List[str]] = None,
    ) -> Dict:
        """Forward pass with different modes.

        Parameters
        ----------
        X : torch.Tensor
            Input observations, shape (B, N, D) or (N, D)
        mode : str
            One of: "train", "intervene", "counterfactual", "plan"
        interventions : dict[str, float], optional
            Interventions for "intervene" or "counterfactual" mode
        target : str or dict[str, float], optional
            Target variable(s) for "counterfactual" or "plan" mode
        interv_nodes : list[str], optional
            Variables to optimize over for "plan" mode

        Returns
        -------
        dict with results depending on mode:
            - "train": {"A": causal_matrix, "loss": total_loss}
            - "intervene": {"state": dict, "E_next": np.ndarray}
            - "counterfactual": {"value": float}
            - "plan": {"a_opt": dict, "energy": float, "history": list}
        """
        if mode == "train":
            return self._forward_train(X)
        elif mode == "intervene":
            return self._forward_intervene(X, interventions)
        elif mode == "counterfactual":
            return self._forward_counterfactual(X, interventions, target)
        elif mode == "plan":
            return self._forward_plan(X, target, interv_nodes)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _forward_train(self, X: torch.Tensor) -> Dict:
        """Training forward pass: learn causal structure from data."""
        # Learn causal matrix
        A, _ = self.causal_ffnn(X)
        self._A = A

        # Contrastive alignment loss: A should capture relationships between variables
        # For embedding data, we can use similarity-based alignment
        X_flat = X if X.dim() == 2 else X.mean(dim=0)  # (N, D)

        # Compute cosine similarity matrix as target (if embeddings are normalized)
        # Note: assumes embeddings are roughly normalized
        with torch.no_grad():
            norms = torch.norm(X_flat, dim=1, keepdim=True)
            X_normed = X_flat / torch.clamp(norms, min=1e-8)
            S = X_normed @ X_normed.T  # (N, N) cosine similarity
            S.fill_diagonal_(0.0)  # no self-similarity

        # Alignment loss: A should correlate with similarity structure
        align_loss = ((A - S) ** 2).mean()

        # Acyclicity constraint
        loss = align_loss
        if self.enforce_acyclic:
            h = acyclicity_loss(A)
            loss = loss + self._lam * h + (self._rho / 2) * h**2

        return {"A": A.detach(), "loss": loss, "align_loss": align_loss.detach()}

    def build_scm(self, X: torch.Tensor) -> NeuralSCM:
        """Build NeuralSCM from learned causal structure.

        This must be called after training before using intervention,
        counterfactual, or planning modes.
        """
        if self._A is None:
            raise RuntimeError(
                "Must train the network first (call forward with mode='train')"
            )

        # Detach and convert to numpy for SCM
        A_np = self._A.detach().cpu().numpy()
        X_np = X.detach().cpu().numpy()

        # Ensure E_raw has 3 dimensions: (n_samples, N, D)
        if X_np.ndim == 2:
            X_np = X_np[np.newaxis, :, :]  # add batch dimension
        elif X_np.ndim == 3:
            # Keep as is
            pass
        else:
            raise ValueError(f"Expected X_np with 2 or 3 dimensions, got {X_np.ndim}")

        # Build SCM with auto-inferred topological order
        self._scm = NeuralSCM.from_embeddings(
            var_names=self.var_names,
            A=A_np,
            E_raw=X_np,
            topo_order=None,  # auto-inferred from A
        )

        # Extract matrices for planner
        self._E = torch.from_numpy(self._scm._E).float()
        self._U = torch.from_numpy(self._scm._U).float()

        return self._scm

    def _ensure_scm_built(self, X: torch.Tensor):
        """Ensure SCM is built, building it if necessary."""
        if self._scm is None:
            logger.info("Building SCM from learned causal structure...")
            self.build_scm(X)

    def _forward_intervene(
        self,
        X: torch.Tensor,
        interventions: Dict[str, float],
    ) -> Dict:
        """Predict intervention effects."""
        self._ensure_scm_built(X)

        # Convert to numpy for SCM
        X_np = X.detach().cpu().numpy()
        if X_np.ndim == 3:
            X_np = X_np.mean(axis=0)

        state, E_next = self._scm.step(interventions, E_init=X_np)
        return {"state": state, "E_next": E_next}

    def _forward_counterfactual(
        self,
        X: torch.Tensor,
        interventions: Dict[str, float],
        target: Union[str, Dict[str, float]],
    ) -> Dict:
        """Counterfactual inference."""
        self._ensure_scm_built(X)

        # Convert to numpy for SCM
        X_np = X.detach().cpu().numpy()
        if X_np.ndim == 3:
            X_np = X_np.mean(axis=0)

        if isinstance(target, str):
            cf_value = self._scm.counterfactual(interventions, target, E_init=X_np)
            return {"value": cf_value}
        else:
            # Multiple targets: compute for each
            results = {}
            for tgt_var in target:
                cf_value = self._scm.counterfactual(interventions, tgt_var, E_init=X_np)
                results[tgt_var] = cf_value
            return {"values": results}

    def _forward_plan(
        self,
        X: torch.Tensor,
        target: Union[str, Dict[str, float]],
        interv_nodes: List[str],
    ) -> Dict:
        """Gradient-based causal planning."""
        self._ensure_scm_built(X)

        # Convert target to dict format
        if isinstance(target, str):
            target_dict = {target: 1.0}  # Default target value
        else:
            target_dict = target

        # Build planner
        if self._planner is None:
            self._planner = CausalPlanner(self._scm)

        # Convert X to numpy for planning
        X_np = X.detach().cpu().numpy()
        if X_np.ndim == 3:
            X_np = X_np.mean(axis=0)

        # Use SCM's E matrix if available, otherwise use input
        E_init = self._scm._E if self._scm._E is not None else X_np

        result = self._planner.plan(
            E_init=E_init,
            target=target_dict,
            interv_nodes=interv_nodes,
        )
        return result

    def update_augmented_lagrangian(self, h_val: float):
        """Update augmented Lagrangian multipliers for acyclicity constraint.

        Call this after each training epoch to tighten the constraint.
        """
        if not self.enforce_acyclic:
            return

        self._lam = self._lam + self._rho * h_val

        # Increase penalty if constraint violation is not decreasing fast enough
        # This is a simple heuristic; can be customized
        if h_val > 0.25 * getattr(self, "_h_prev", float("inf")):
            self._rho = min(self._rho * 10.0, 1e6)

        self._h_prev = h_val

    def get_causal_matrix(self) -> Optional[np.ndarray]:
        """Get learned causal weight matrix."""
        if self._A is None:
            return None
        return self._A.detach().cpu().numpy()

    def get_topological_order(self) -> Optional[List[str]]:
        """Get topological order from learned causal matrix."""
        A_np = self.get_causal_matrix()
        if A_np is None:
            return None
        return topo_order_from_A(A_np, self.var_names)

    def to(self, device):
        """Move network to device."""
        super().to(device)
        if self._E is not None:
            self._E = self._E.to(device)
        if self._U is not None:
            self._U = self._U.to(device)
        return self
