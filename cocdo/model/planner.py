"""CausalPlanner — gradient-based action optimizer for NeuralSCM world model.

Implements LeCun-style planning:
    a* = argmin_a  E[ predict(s_t, a),  s_target ]

where:
    predict(s_t, a)  =  A_do^T @ E_do + U       (causal rollout in embedding space)
    E[·, ·]          =  Σ (||E_next[j]|| - target_j)²  (norm-based energy)

Energy compares embedding norms rather than full vectors, avoiding direction
misalignment between the target and propagated embeddings.  The energy is
fully differentiable w.r.t. the intervention values a, so we can use Adam to
find the optimal causal action without RL sampling.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from .scm import NeuralSCM

logger = logging.getLogger(__name__)


class CausalPlanner:
    """Gradient-based causal action optimizer.

    Parameters
    ----------
    scm : NeuralSCM
        A built SCM (must have _A, _E, _U matrices).
    """

    def __init__(self, scm: NeuralSCM):
        if scm._A is None or scm._E is None or scm._U is None:
            raise RuntimeError(
                "CausalPlanner requires A, E, U matrices. Call CocDo.build() first."
            )
        self.scm = scm
        # Pre-convert static matrices to torch (no grad)
        self._A = torch.from_numpy(scm._A).float()   # (N, N)
        self._U = torch.from_numpy(scm._U).float()   # (N, D)

    # ── Core differentiable operations ────────────────────────────────────────

    def _make_col_mask(self, interv_nodes: list[str], N: int) -> torch.Tensor:
        """Binary column mask: col_mask[:, k] = 1 for each intervened node k."""
        col_mask = torch.zeros(N, N)
        for name in interv_nodes:
            k = self.scm._var_names.index(name)
            col_mask[:, k] = 1.0
        return col_mask  # (N, N), no grad

    def _target_embedding(
        self,
        target: dict[str, float],
    ) -> tuple[list[int], torch.Tensor]:
        """Convert {var: scalar} target to (target_indices, scalar_targets tensor)."""
        target_idx = []
        scalars = []
        for name, scalar in target.items():
            target_idx.append(self.scm._var_names.index(name))
            scalars.append(scalar)
        return target_idx, torch.tensor(scalars, dtype=torch.float32)

    def _step(
        self,
        a: torch.Tensor,            # (K,) intervention values, requires_grad
        interv_nodes: list[str],
        E_cur: torch.Tensor,        # (N, D) current embedding, no grad
        cut_incoming: bool = True,
    ) -> torch.Tensor:
        """One differentiable causal propagation step under intervention a.

        cut_incoming=True  (do-calculus): zero columns of intervened nodes in A.
            Use when interv_nodes ≠ target_nodes (classic X→Y planning).
        cut_incoming=False (soft substitution): keep full A.
            Use when interv_nodes == target_nodes (e.g. DoLM gen embedding search),
            so that E_next[interv] still depends on a via A.T[interv,:] @ E_do.

        Returns E_next (N, D), differentiable w.r.t. a.
        """
        N = E_cur.shape[0]
        if cut_incoming:
            col_mask = self._make_col_mask(interv_nodes, N)
            A_do = self._A * (1.0 - col_mask)
        else:
            A_do = self._A

        # Build E_do differentiably: replace intervened rows with direction * a[i].
        one_hots = torch.zeros(len(interv_nodes), N)
        directions = []
        for i, name in enumerate(interv_nodes):
            k = self.scm._var_names.index(name)
            one_hots[i, k] = 1.0
            e_k = E_cur[k]
            directions.append(e_k / e_k.norm().clamp(min=1e-8))

        # interv_E: (N, D) — weighted sum of interventions, grad flows through a
        interv_E = sum(
            one_hots[i].unsqueeze(1) * directions[i].unsqueeze(0) * a[i]
            for i in range(len(interv_nodes))
        )

        # row_mask: (N, 1) — 1 at each intervened row
        row_mask = one_hots.sum(0).clamp(max=1.0).unsqueeze(1)
        E_do = (1.0 - row_mask) * E_cur + row_mask * interv_E   # (N, D)

        return A_do.T @ E_do + self._U                          # (N, D)

    def energy(
        self,
        a: torch.Tensor,
        interv_nodes: list[str],
        E_init: torch.Tensor,
        target_idx: list[int],
        scalar_targets: torch.Tensor,
        rollout_steps: int = 1,
        cut_incoming: bool = True,
    ) -> torch.Tensor:
        """Compute inconsistency energy E[predict(s_t, a), s_target].

        Fully differentiable w.r.t. a.

        Returns scalar energy = Σ (||E_next[j]|| - target_scalar[j])².
        Comparing embedding norms rather than full vectors avoids direction
        misalignment between E_target and the propagated E_next.
        """
        E_cur = E_init
        for _ in range(rollout_steps):
            E_cur = self._step(a, interv_nodes, E_cur, cut_incoming=cut_incoming)
        predicted_norms = E_cur[target_idx].norm(dim=-1)   # (|targets|,)
        return ((predicted_norms - scalar_targets) ** 2).sum()

    # ── Planning API ──────────────────────────────────────────────────────────

    def plan(
        self,
        E_init: np.ndarray,
        target: dict[str, float],
        interv_nodes: list[str],
        a_init: Optional[np.ndarray] = None,
        lr: float = 0.05,
        steps: int = 200,
        rollout_steps: int = 1,
        cut_incoming: bool = True,
        verbose: bool = False,
    ) -> dict:
        """Find optimal causal intervention via gradient descent on the energy.

        Parameters
        ----------
        E_init        : (N, D) current embedding state (e.g. scm._E)
        target        : desired scalar values per node, e.g. {'revenue': 1.5}
        interv_nodes  : which variables to intervene on, e.g. ['ad_spend']
        a_init        : initial guess for intervention values (zeros by default)
        lr            : Adam learning rate
        steps         : optimisation steps
        rollout_steps : causal propagation steps to unroll per optimisation step.
                        Use >1 when the causal path from interv_nodes to target
                        is longer than 1 hop (e.g. X→Y→Z needs rollout_steps=2).
        verbose       : log energy every 50 optimisation steps

        Returns
        -------
        dict with:
          'a_opt'   : {node_name: optimal_value}  — the optimal action
          'energy'  : final energy value
          'history' : list of energy values per optimisation step
        """
        K = len(interv_nodes)
        if a_init is None:
            a_init = np.zeros(K, dtype=np.float32)

        E_t = torch.from_numpy(E_init).float()
        target_idx, scalar_targets = self._target_embedding(target)

        a = torch.tensor(a_init, dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([a], lr=lr)

        history: list[float] = []
        for step in range(steps):
            opt.zero_grad()
            e = self.energy(a, interv_nodes, E_t, target_idx, scalar_targets,
                            rollout_steps=rollout_steps, cut_incoming=cut_incoming)
            e.backward()
            opt.step()
            e_val = float(e.detach())
            history.append(e_val)
            if verbose and (step % 50 == 0 or step == steps - 1):
                logger.info("step %d  energy=%.6f  a=%s", step, e_val,
                            [f"{v:.4f}" for v in a.detach().tolist()])

        a_opt_vals = a.detach().tolist()
        return {
            "a_opt":   {name: a_opt_vals[i] for i, name in enumerate(interv_nodes)},
            "energy":  history[-1],
            "history": history,
        }
