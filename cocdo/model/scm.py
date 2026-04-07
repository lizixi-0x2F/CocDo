"""NeuralSCM — COC-encoded Structural Causal Model with do-calculus support."""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..kernel.terms import Sort, Var, Const, Pi, Lam, App, Term, Add
from ..kernel.reduction import beta_reduce, subst
from ..kernel.typing import Context, type_of, check_intervention

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
    A         : (N, N) causal weight matrix from CausalFFNN            [optional]
    E         : (N, D) trained token embeddings                 [optional]
    U         : (N, D) residuals U_j = E_j - A^T E[j]          [optional]
    """

    def __init__(
        self,
        var_names: list[str],
        A: Optional[np.ndarray] = None,
        E: Optional[np.ndarray] = None,
        U: Optional[np.ndarray] = None,
        topo_order: Optional[list[str]] = None,
    ):
        self.vars: dict[str, CausalNode] = {}
        self.coc_context: Context = {}
        self._var_names = list(var_names)   # index → name
        self._A = A   # (N, N) or None
        self._E = E   # (N, D) or None
        self._U = U   # (N, D) or None

        # topo_order is a list of var names in topological order (roots first).
        # Position in topo_order becomes the Sort level: topo_order[0] → Sort(0),
        # topo_order[1] → Sort(1), etc.  Unknown names default to Sort(0).
        if topo_order is not None:
            level_of = {name: i for i, name in enumerate(topo_order)}
        else:
            level_of = {}

        for name in var_names:
            level = level_of.get(name, 0)
            node_type = Sort(level)
            self.vars[name] = CausalNode(name=name, coc_type=node_type)
            self.coc_context[name] = node_type
            self.coc_context[f"val_{name}"] = Var(name)

    @classmethod
    def from_embeddings(
        cls,
        var_names: list[str],
        A: np.ndarray,
        E_raw: np.ndarray,
        topo_order: Optional[list[str]] = None,
        edge_threshold: float = 1e-4,
    ) -> "NeuralSCM":
        """Standard build path: construct SCM from raw sample embeddings.

        Computes RMS embeddings (rather than mean) so that each node's
        embedding has a stable non-zero norm — required for CausalPlanner
        to have well-defined intervention directions.

        Parameters
        ----------
        var_names     : node names, length N
        A             : (N, N) causal weight matrix from CausalFFNN
        E_raw         : (n_samples, N, D) per-sample embeddings
        topo_order    : node names in topological order (roots first).
                        If None, defaults to var_names order.
        edge_threshold: minimum A[i,j] weight to register an edge

        Returns
        -------
        NeuralSCM with all edges added and matrices set.
        """
        # RMS embedding: captures the typical magnitude of each node's embedding
        # without cancellation from zero-mean distributions.
        E = np.sqrt((E_raw ** 2).mean(axis=0))   # (N, D)
        U = E - A.T @ E                           # exogenous residual

        scm = cls(var_names=var_names, A=A, E=E, U=U, topo_order=topo_order)

        order = topo_order if topo_order is not None else var_names
        for i, p in enumerate(order):
            for j, c in enumerate(order):
                if i < j:
                    pi = var_names.index(p)
                    ci = var_names.index(c)
                    w  = float(A[pi, ci])
                    if w > edge_threshold:
                        scm.add_causal_edge(p, c, weight=w)

        return scm

    def add_causal_edge(self, parent: str, child: str, weight: float = 1.0):
        """
        Add a soft causal edge parent → child with continuous weight.

        The weight (from CausalFFNN's SiLU output) is encoded as a COC Const
        carrying the causal strength, so the mechanism type becomes:
            f_Y : Pi(w_X : Const(weight), Pi(X : Type0, ... Type0))
        """
        if child not in self.vars:
            raise ValueError(f"Variable {child!r} not defined")
        if parent not in self.vars:
            raise ValueError(f"Variable {parent!r} not defined")

        p_type = self.vars[parent].coc_type
        c_type = self.vars[child].coc_type
        if not isinstance(p_type, Sort) or not isinstance(c_type, Sort):
            raise TypeError(
                f"Topological guard requires Sort types; got {p_type!r} and {c_type!r}"
            )
        parent_level = p_type.level
        child_level  = c_type.level
        if parent_level >= child_level:
            raise TypeError(
                f"Causal edge {parent!r}→{child!r} violates topological order: "
                f"Sort({parent_level}) is not strictly less than Sort({child_level}). "
                f"This would introduce a cycle or a backward edge."
            )

        node = self.vars[child]
        if parent not in node.parents:
            node.parents.append(parent)
        node.parent_weights[parent] = weight

        # Register weight constant in context.
        self.coc_context[f"w_{parent}->{child}"] = Sort(0)

        # Rebuild mechanism as a flat weighted sum over all current parents:
        #   body = Add(w_p1 * Var(p1), Add(w_p2 * Var(p2), ...))
        # Wrapped in a curried Lam for each parent, then applied to Var(parent)
        # so that subst(open_term, p_i, Const(do_p_i, v)) correctly replaces
        # every Var(p_i) in the body regardless of how many parents exist.
        #
        # After do(p_k = v):
        #   subst replaces Var(p_k) → Const("do_pk", v) everywhere in open_term
        #   beta_reduce eliminates the Lam(p_k, ...) / App(..., Const) pair
        #   remaining parents stay as free Var, awaiting their own subst
        parents = node.parents
        weights = node.parent_weights

        # body: weighted sum  Σ w_i * Var(p_i)
        body: Term = App(Const(f"w_{parents[0]}->{child}", weights[parents[0]]), Var(parents[0]))
        for p in parents[1:]:
            term_p = App(Const(f"w_{p}->{child}", weights[p]), Var(p))
            body = App(App(Add, body), term_p)

        # curried Lam: Lam(p1, Sort(0), Lam(p2, Sort(0), ... body))
        mech: Term = body
        for p in reversed(parents):
            mech = Lam(p, Sort(0), mech)

        # open term: App(...App(mech, Var(p1))..., Var(pN))
        open_term: Term = mech
        for p in parents:
            open_term = App(open_term, Var(p))

        # Type annotation in context: Pi-chain over all parents → Sort(0)
        mech_pi_type: Term = self.vars[child].coc_type
        for p in parents:
            mech_pi_type = Pi(p, Sort(0), mech_pi_type)

        mech_name = f"f_{child}"
        self.coc_context[mech_name] = mech_pi_type
        node.mechanism = open_term
        logger.debug("mechanism: %s := %s", mech_name, open_term)

    def observe(self, var: str, value: float):
        self.vars[var].observed_value = value
        self.coc_context[f"obs_{var}"] = self.vars[var].coc_type

    def do(self, var: str, value: float) -> "NeuralSCM":
        """
        Intervention do(var = value).

        1. CoC guard: type-checks the intervention before any state is modified
        2. Remove mechanism f_{var} (sever incoming edges)
        3. Fix var as a constant term
        4. Substitute into all descendant open Lam+App mechanisms and beta-reduce
        Returns a new SCM; original is unchanged.
        """
        logger.info("Intervention: do(%s = %.3f)", var, value)

        # CoC guard — runs BEFORE copying state or touching numpy.
        # Raises TypeError if var is undeclared or type-inconsistent.
        check_intervention(var, value, self.coc_context)

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
                if child_node.mechanism is not None:
                    # Substitute into the open TERM (Lam+App), not the Pi type.
                    # subst replaces Var(var) in the App arg, then beta_reduce fires.
                    old_open = child_node.mechanism
                    new_open = beta_reduce(subst(old_open, var, const_term))
                    child_node.mechanism = new_open
                    # Also update the Pi type annotation to remove the intervened parent
                    if mech_key in intervened.coc_context:
                        old_pi = intervened.coc_context[mech_key]
                        new_pi = beta_reduce(subst(old_pi, var, const_term))
                        intervened.coc_context[mech_key] = new_pi
                    child_node.parents = [p for p in child_node.parents if p != var]
                    child_node.parent_weights.pop(var, None)
                    logger.debug("do(%s): %s reduced to %s", var, mech_key, new_open)

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
            result  = float(np.linalg.norm(E_j_cf))
        elif node.observed_value is not None:
            result = node.observed_value
        elif not node.parents:
            result = 0.0
        else:
            parent_vals = [self.vars[p].observed_value or 0.0 for p in node.parents]
            result = float(np.mean(parent_vals))

        # Write result back into CoC context as a grounded Const term.
        # val_{target} changes from Var(target) to Const("obs_{target}", result),
        # making the inferred value a first-class term for downstream queries.
        node.observed_value = result
        self.coc_context[f"obs_{target}"] = node.coc_type
        self.coc_context[f"val_{target}"] = Const(f"obs_{target}", result)
        return result

    def step(
        self,
        interventions: dict[str, float],
        E_init: Optional[np.ndarray] = None,
    ) -> tuple[dict[str, float], np.ndarray]:
        """
        Apply interventions and return (state_dict, E_next).

        E_next is the full (N, D) embedding matrix after propagation —
        pass it as E_init to the next step for multi-step rollout.

        Parameters
        ----------
        interventions : {var_name: scalar_value}
        E_init        : (N, D) starting embedding; defaults to self._E

        Returns
        -------
        state : {var_name: scalar}  — scalar value per node
        E_next: (N, D)             — updated embedding matrix
        """
        if self._A is None or self._E is None or self._U is None:
            raise RuntimeError(
                "step() requires A, E, U matrices. Call CocNN.build() first."
            )
        E_cur = self._E if E_init is None else E_init

        # ── Action: build A_do and E_do ────────────────────────────────────
        A_do = self._A.copy()
        E_do = E_cur.copy()

        for var_name, scalar_val in interventions.items():
            if var_name not in self._var_names:
                raise ValueError(f"Unknown variable {var_name!r}")
            k = self._var_names.index(var_name)
            A_do[:, k] = 0.0
            e_k = E_cur[k]
            e_k_norm = float(np.linalg.norm(e_k))
            if e_k_norm > 1e-8:
                E_do[k] = e_k / e_k_norm * scalar_val
            else:
                E_do[k] = np.zeros_like(e_k)

        # ── Propagate all nodes at once: E_next = A_do^T @ E_do + U ────────
        E_next = A_do.T @ E_do + self._U   # (N,N)^T @ (N,D) = (N,D)

        # Intervened nodes keep their fixed embedding (do severs the equation)
        for var_name, scalar_val in interventions.items():
            k = self._var_names.index(var_name)
            E_next[k] = E_do[k]

        state: dict[str, float] = {}
        for name in self._var_names:
            if name in interventions:
                state[name] = interventions[name]
            else:
                j = self._var_names.index(name)
                state[name] = float(np.linalg.norm(E_next[j]))

        return state, E_next

    def rollout(
        self,
        action_sequence: list[dict[str, float]],
        reward_fn: "Callable[[dict[str, float], int], float]",
        E_init: Optional[np.ndarray] = None,
        discount: float = 1.0,
    ) -> tuple[list[dict[str, float]], float]:
        """
        H-step causal rollout under a sequence of interventions.

        Each step's output embedding E_next feeds directly into the next
        step as E_init, so causal effects propagate across time.

        Parameters
        ----------
        action_sequence : list of H intervention dicts, one per time step
        reward_fn       : r(state, t) → float  — scalar reward at step t
        E_init          : (N, D) starting embedding; defaults to self._E
        discount        : γ ∈ (0, 1] — discount factor for cumulative reward

        Returns
        -------
        trajectory     : list of H state dicts
        total_reward   : discounted sum of rewards
        """
        if self._A is None or self._E is None or self._U is None:
            raise RuntimeError(
                "rollout() requires A, E, U matrices. Call CocNN.build() first."
            )
        E_cur = (self._E if E_init is None else E_init).copy()
        trajectory: list[dict[str, float]] = []
        total_reward = 0.0

        for t, interventions in enumerate(action_sequence):
            state, E_cur = self.step(interventions, E_init=E_cur)
            trajectory.append(state)
            total_reward += (discount ** t) * reward_fn(state, t)

        return trajectory, total_reward

    def counterfactual(
        self,
        interventions: dict[str, float],
        target: str,
        E_init: Optional[np.ndarray] = None,
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
        target        : variable whose counterfactual value to predict
        E_init        : (N, D) starting embedding; defaults to self._E

        Returns
        -------
        Counterfactual scalar value of `target`.
        """
        if self._A is None or self._E is None or self._U is None:
            raise RuntimeError(
                "counterfactual() requires A, E, U matrices. "
                "Pass them when constructing NeuralSCM or call CocNN.build()."
            )

        E_cur = self._E if E_init is None else E_init
        j = self._var_names.index(target)

        # ── Step 1: Abduction — U is already known ─────────────────────────
        U_j = self._U[j]   # (D,)  exogenous noise for target token

        # ── Step 2: Action — modify A and E ────────────────────────────────
        A_do = self._A.copy()
        E_do = E_cur.copy()

        for var_name, scalar_val in interventions.items():
            if var_name not in self._var_names:
                raise ValueError(f"Unknown variable {var_name!r}")
            k = self._var_names.index(var_name)
            A_do[:, k] = 0.0
            e_k = E_cur[k]
            e_k_norm = float(np.linalg.norm(e_k))
            if e_k_norm > 1e-8:
                E_do[k] = e_k / e_k_norm * scalar_val
            else:
                E_do[k] = np.zeros_like(e_k)

        # ── Step 3: Prediction ──────────────────────────────────────────────
        E_j_star = A_do[:, j] @ E_do + U_j
        return float(np.linalg.norm(E_j_star))

