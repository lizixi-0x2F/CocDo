# CocDo

**A minimal causal world model — learn causal graphs, intervene, plan.**

CocDo learns a causal graph from observations, then lets you run `do(X=v)` interventions, counterfactuals, and gradient-based planning — all type-checked by a lightweight COC kernel that rejects cycles and inconsistent queries at build time.

```python
from cocdo import CausalFFNN, NeuralSCM
from cocdo.model import CausalPlanner
import torch

# 1. Learn causal weights
ffnn = CausalFFNN(d_embed=32)
# ... training loop: loss = ((X @ A - X)**2).mean() ...

# 2. Build the causal world model (one call)
with torch.no_grad():
    A, _ = ffnn(E_raw)
scm = NeuralSCM.from_embeddings(
    var_names  = ["ad_spend", "clicks", "revenue"],
    A          = A.numpy(),
    E_raw      = E_raw.numpy(),
    topo_order = ["ad_spend", "clicks", "revenue"],
)

# 3. Intervene
state, _ = scm.step({"ad_spend": 3.0})
print(state["revenue"])          # causal effect, not correlation

# 4. Counterfactual  (Pearl layer 3)
cf = scm.counterfactual({"ad_spend": 3.0}, target="revenue")

# 5. Gradient planning — find the optimal action, no sampling
planner = CausalPlanner(scm)
result  = planner.plan(
    E_init       = scm._E,
    target       = {"revenue": 2.5},
    interv_nodes = ["ad_spend"],
)
print(result["a_opt"])           # {"ad_spend": 1.84}
```

---

## How it works

```
Observations  (N samples, N vars)
    ↓  CausalFFNN        pairwise edge scorer → sigmoid A ∈ R^{N×N}
    ↓  from_embeddings   RMS embeddings + exogenous residuals U
    ↓  NeuralSCM         COC type-checks the graph, rejects cycles
    ↓  do() / step()     Pearl do-calculus via term substitution + β-reduction
    ↓  counterfactual()  Abduction → Action → Prediction
    ↓  CausalPlanner     Adam on ‖‖E_next[j]‖ − target_j‖² — no RL needed
```

**COC type guard** — every edge `X → Y` is encoded as a dependent Pi-type `Π(X : Typeᵢ). Typeⱼ` where `i < j` follows topological order. Any edge that would introduce a cycle raises `TypeError` before touching any matrix.

**Norm-based energy** — the planner minimises `Σ (‖E_next[j]‖ − target_j)²`. Comparing embedding norms rather than full vectors eliminates direction-misalignment, so the energy is exactly zero at the true solution.

---

## Install

```bash
git clone https://github.com/lizixi-0x2F/CocDo
cd CocDo
pip install -e .
```

Requires Python ≥ 3.10, PyTorch ≥ 2.0, NumPy ≥ 1.24.

---

## API

### `CausalFFNN`

Learns the causal weight matrix from embeddings.

```python
ffnn = CausalFFNN(d_embed, hidden=256)
A, logits = ffnn(E)   # E: (B, N, D) or (N, D)  →  A: (N, N)
```

Each entry `A[i,j] ∈ (0,1)` is a sigmoid gate — the causal strength of edge `i → j`. Diagonal is forced to zero (no self-loops).

**Training loss** — structural equation supervised on scalar observations:
```python
loss = ((X @ A - X) ** 2).mean()   # X: (B, N) scalar observations
```

### `NeuralSCM`

The causal world model. Build it with `from_embeddings`:

```python
scm = NeuralSCM.from_embeddings(
    var_names,       # list of N node names
    A,               # (N, N) from CausalFFNN
    E_raw,           # (n_samples, N, D) per-sample embeddings
    topo_order,      # node names sorted root-first
    edge_threshold,  # minimum A[i,j] to register an edge (default 1e-4)
)
```

Internally computes RMS embeddings `E = sqrt(mean(E_raw²))` and residuals `U = E − A^T E`, then adds all edges respecting topological order.

| Method | Description |
|--------|-------------|
| `do(var, value) → NeuralSCM` | Intervention — severs incoming edges, returns new SCM |
| `step(interventions, E_init)` | One-step causal propagation: `E_next = A_do^T E_do + U` |
| `rollout(action_sequence, reward_fn, discount)` | H-step rollout with discounted reward |
| `counterfactual(interventions, target)` | Pearl 3-step: abduction → action → prediction |
| `infer_effect(target) → float` | Predict target scalar from structural equation |

### `CausalPlanner`

Gradient-based optimal intervention search.

```python
planner = CausalPlanner(scm)
result  = planner.plan(
    E_init,          # scm._E  (RMS embeddings)
    target,          # {var_name: desired_scalar}
    interv_nodes,    # variables to optimise over
    lr=0.05,
    steps=200,
    rollout_steps=1, # increase for multi-hop paths (X→Y→Z needs 2)
)
# result: {"a_opt": {name: value}, "energy": float, "history": [...]}
```

Energy function (exact zero at solution):
```
a* = argmin_a  Σ_j (‖E_next[j]‖ − target_j)²
```

### `NodeProjector`

Projects node embeddings to scalar observations.

```python
proj = NodeProjector(d_embed)
y = proj(E)   # (N, D) → (N,)
```

---

## COC kernel

```
cocdo/kernel/
├── terms.py      # AST: Sort, Var, Const, Pi, Lam, App
├── reduction.py  # capture-avoiding substitution + call-by-value β-reduction
└── typing.py     # type_of(), check_intervention()
```

`do(X = v)` is implemented as `subst(mechanism, X, Const(v))` followed by `beta_reduce`. The type checker runs before any numpy computation — a failed check leaves the SCM untouched.

---

## gCastle benchmark

Validated against synthetic DAGs from [gCastle](https://github.com/huawei-noah/trustworthyAI/tree/master/gcastle):

```bash
pip install gcastle
python examples/demo_gcastle.py
```

```
ground-truth edges:
  x0 → x1  x0 → x2  x0 → x3
  x1 → x3  x2 → x1  x2 → x3

CocDo learned A (strongest edges):
  x0 → x3  (w=0.974)    x1 → x3  (w=1.000)    x2 → x3  (w=0.988)
  x0 → x1  (w=0.979)    x2 → x1  (w=0.987)

Planner — find x0 s.t. x3 reaches do(x0=3.0) value:
  optimal x0 = 3.0000   energy = 0.000000  ✓
```

---

## Repository layout

```
cocdo/
├── kernel/
│   ├── terms.py        COC AST nodes
│   ├── reduction.py    β-reduction + substitution
│   └── typing.py       type checker + intervention guard
└── model/
    ├── causal_ffnn.py  CausalFFNN, NodeProjector
    ├── scm.py          NeuralSCM — do(), rollout(), counterfactual(), from_embeddings()
    └── planner.py      CausalPlanner — norm-based energy, Adam planning
examples/
└── demo_gcastle.py     gCastle synthetic DAG → full pipeline
```

---

## Citation

```bibtex
@software{cocdo2026,
  author = {lizixi},
  title  = {CocDo: A Minimal Causal World Model with COC Type-Theoretic Interventions},
  year   = {2026},
  url    = {https://github.com/lizixi-0x2F/CocDo}
}
```
