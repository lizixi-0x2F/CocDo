"""NeuralSCM — COC-encoded Structural Causal Model with do-calculus support."""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..kernel.terms import Sort, Var, Const, Pi, Term
from ..kernel.reduction import beta_reduce, subst
from ..kernel.typing import Context

logger = logging.getLogger(__name__)


@dataclass
class CausalNode:
    name: str
    coc_type: Term
    parents: list[str] = field(default_factory=list)
    parent_weights: dict[str, float] = field(default_factory=dict)
    mechanism: Optional[Term] = None
    observed_value: Optional[float] = None


class NeuralSCM:
    """
    Neural Structural Causal Model.

    Encodes a causal graph as COC dependent types.
    do(X=a) is implemented as COC term substitution + beta reduction.
    The type checker intercepts logically inconsistent queries.

    Structural equations (Pearl layer 2)
    -------------------------------------
    Each node j: E_j = A[:,j] @ E + U_j
    where A is the learned causal weight matrix and U_j is the exogenous residual.

    Parameters
    ----------
    var_names : token node names (length N)
    A         : (N, N) causal weight matrix from SeqCausalFFNN  [optional]
    E         : (N, D) trained token embeddings                 [optional]
    U         : (N, D) residuals U_j = E_j - A^T E[j]          [optional]
    """

    def __init__(
        self,
        var_names: list[str],
        A: Optional[np.ndarray] = None,
        E: Optional[np.ndarray] = None,
        U: Optional[np.ndarray] = None,
    ):
        self.vars: dict[str, CausalNode] = {}
        self.coc_context: Context = {}
        self._var_names = list(var_names)   # index → name
        self._A = A   # (N, N) or None
        self._E = E   # (N, D) or None
        self._U = U   # (N, D) or None

        for name in var_names:
            node_type = Sort(0)
            self.vars[name] = CausalNode(name=name, coc_type=node_type)
            self.coc_context[name] = node_type
            self.coc_context[f"val_{name}"] = Var(name)

    def add_causal_edge(self, parent: str, child: str, weight: float = 1.0):
        """
        Add a soft causal edge parent → child with continuous weight.

        The weight (from softmax) is encoded as a COC Const carrying the
        causal strength, so the mechanism type becomes:
            f_Y : Pi(w_X : Const(weight), Pi(X : Type0, ... Type0))
        """
        if child not in self.vars:
            raise ValueError(f"Variable {child!r} not defined")
        node = self.vars[child]
        if parent not in node.parents:
            node.parents.append(parent)
        node.parent_weights[parent] = weight

        # Build mechanism type: weighted Pi-chain
        # Each parent contributes a Pi with a Const weight annotation
        mech_type = self.vars[child].coc_type
        for p in reversed(node.parents):
            w = node.parent_weights[p]
            # Annotate the domain with causal strength via Const
            weighted_domain = Const(f"w_{p}->{child}", w)
            self.coc_context[f"w_{p}->{child}"] = Sort(0)
            mech_type = Pi(p, weighted_domain, mech_type)

        mech_name = f"f_{child}"
        self.coc_context[mech_name] = mech_type
        node.mechanism = mech_type
        logger.debug("mechanism: %s : %s", mech_name, mech_type)

    def observe(self, var: str, value: float):
        self.vars[var].observed_value = value
        self.coc_context[f"obs_{var}"] = self.vars[var].coc_type

    def do(self, var: str, value: float) -> "NeuralSCM":
        """
        Intervention do(var = value).

        1. Remove mechanism f_{var} (sever incoming edges)
        2. Fix var as a constant term
        3. Substitute into all descendant mechanisms and beta-reduce
        Returns a new SCM; original is unchanged.
        """
        logger.info("Intervention: do(%s = %.3f)", var, value)

        intervened = NeuralSCM.__new__(NeuralSCM)
        intervened.vars = {k: CausalNode(
            name=v.name, coc_type=v.coc_type,
            parents=list(v.parents), parent_weights=dict(v.parent_weights),
            mechanism=v.mechanism,
            observed_value=v.observed_value
        ) for k, v in self.vars.items()}
        intervened.coc_context  = dict(self.coc_context)
        intervened._var_names   = list(self._var_names)
        intervened._A           = self._A.copy() if self._A is not None else None
        intervened._E           = self._E.copy() if self._E is not None else None
        intervened._U           = self._U.copy() if self._U is not None else None

        mech_name = f"f_{var}"
        if mech_name in intervened.coc_context:
            del intervened.coc_context[mech_name]
            logger.debug("deleted mechanism: %s", mech_name)
        intervened.vars[var].parents  = []
        intervened.vars[var].parent_weights = {}
        intervened.vars[var].mechanism = None

        const_term = Const(f"do_{var}", value)
        # Register do_{var} so type_of() can resolve it in substituted mechanisms
        intervened.coc_context[f"do_{var}"] = intervened.vars[var].coc_type
        intervened.coc_context[f"obs_{var}"] = intervened.vars[var].coc_type
        intervened.vars[var].observed_value = value

        for child_name, child_node in intervened.vars.items():
            if var in child_node.parents:
                mech_key = f"f_{child_name}"
                if mech_key in intervened.coc_context:
                    old_mech = intervened.coc_context[mech_key]
                    new_mech = beta_reduce(subst(old_mech, var, const_term))
                    intervened.coc_context[mech_key] = new_mech
                    child_node.parents = [p for p in child_node.parents if p != var]
                    child_node.parent_weights.pop(var, None)
                    logger.debug("updated %s: %s -> %s", mech_key, old_mech, new_mech)

        return intervened

    def infer_effect(self, target: str) -> float:
        """
        Predict the embedding norm of `target` using the structural equation.

        Structural equation:  E_j = A[:,j] @ E + U_j
        We return ||A[:,j] @ E + U_j||₂ as a scalar proxy for the node's value.

        Falls back to observed_value (embedding norm stored by explainer) when
        A / E / U are not available (e.g. loaded from an old cache).
        """
        node = self.vars[target]
        j    = self._var_names.index(target)

        if self._A is not None and self._E is not None and self._U is not None:
            # True structural equation: weighted sum of parent embeddings + noise
            E_j_hat = self._A[:, j] @ self._E   # (D,)
            E_j_cf  = E_j_hat + self._U[j]       # add back exogenous noise
            return float(np.linalg.norm(E_j_cf))

        # Fallback: use observed embedding norm or mean-of-parents heuristic
        if node.observed_value is not None:
            return node.observed_value
        if not node.parents:
            return 0.0
        parent_vals = [self.vars[p].observed_value or 0.0 for p in node.parents]
        return float(np.mean(parent_vals))

    def counterfactual(
        self,
        interventions: dict[str, float],
        target: str,
    ) -> float:
        """
        Pearl three-step counterfactual inference.

        Step 1 — Abduction
            U_j = E_j - A[:,j] @ E   (already stored as self._U)
            The exogenous noise is pinned to its observed value.

        Step 2 — Action
            For each intervention do(X_k = v_k):
              - zero out column k of A (sever all incoming edges)
              - set E[k] to a vector whose norm equals v_k (scale the unit vector)

        Step 3 — Prediction
            E_j* = A_do[:,j] @ E_do + U_j
            Return ||E_j*||₂ as scalar.

        Parameters
        ----------
        interventions : {token_name: scalar_value}
            e.g. {"t3_Berlin": 0.0}  sets that token's embedding norm to 0
        target : token name whose counterfactual value to predict

        Returns
        -------
        Counterfactual embedding norm of `target`.
        """
        if self._A is None or self._E is None or self._U is None:
            raise RuntimeError(
                "counterfactual() requires A, E, U matrices. "
                "Pass them when constructing NeuralSCM or call CocNN.build()."
            )

        j = self._var_names.index(target)

        # ── Step 1: Abduction — U is already known ─────────────────────────
        U_j = self._U[j]   # (D,)  exogenous noise for target token

        # ── Step 2: Action — modify A and E ────────────────────────────────
        A_do = self._A.copy()   # (N, N)
        E_do = self._E.copy()   # (N, D)

        for var_name, scalar_val in interventions.items():
            if var_name not in self._var_names:
                raise ValueError(f"Unknown variable {var_name!r}")
            k = self._var_names.index(var_name)

            # Sever all incoming edges to the intervened node (column k → 0)
            A_do[:, k] = 0.0

            # Replace embedding of intervened node with a scaled version
            # preserving direction but setting norm to scalar_val
            e_k     = self._E[k]
            e_k_norm = float(np.linalg.norm(e_k))
            if e_k_norm > 1e-8:
                E_do[k] = e_k / e_k_norm * scalar_val
            else:
                E_do[k] = np.zeros_like(e_k)

        # ── Step 3: Prediction ──────────────────────────────────────────────
        E_j_star = A_do[:, j] @ E_do + U_j   # structural equation + noise
        return float(np.linalg.norm(E_j_star))

