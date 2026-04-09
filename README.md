# CocDo

**A minimal causal world model — learn causal graphs, intervene, plan.**

CocDo fuses two ideas that rarely appear together:

- **Pearl's do-calculus** — `do(X=v)` severs incoming edges and propagates effects through a structural causal model
- **COC type theory** — every edge `X → Y` is encoded as a dependent Pi-type `Π(X : Typeᵢ). Typeⱼ` where `i < j` follows topological order; cycles are structurally impossible to express

The result: a differentiable SCM where planning is gradient descent, not RL, and interventions are literally lambda-calculus term substitution.

```python
from cocdo import CausalFFNN, NeuralSCM
from cocdo.model import CausalPlanner
import torch

# 1. Learn causal weights from embeddings (NOTEARS acyclicity enforced during training)
ffnn = CausalFFNN(d_embed=32)
# loss = recon + λ·h(A) + ρ/2·h(A)²   where h(A) = tr(e^{A∘A}) - n

# 2. Build the causal world model — topo_order inferred automatically from A
with torch.no_grad():
    A, _ = ffnn(E_raw)
scm = NeuralSCM.from_embeddings(
    var_names = ["ad_spend", "clicks", "revenue"],
    A         = A.numpy(),
    E_raw     = E_raw.numpy(),
    # topo_order omitted — auto-extracted via Kahn's algorithm
)

# 3. Intervene — causal effect, not correlation
state, _ = scm.step({"ad_spend": 3.0})

# 4. Counterfactual  (Pearl layer 3)
cf = scm.counterfactual({"ad_spend": 3.0}, target="revenue")

# 5. Gradient planning — no sampling, no RL
planner = CausalPlanner(scm)
result  = planner.plan(
    E_init       = scm._E,
    target       = {"revenue": 2.5},
    interv_nodes = ["ad_spend"],
)
print(result["a_opt"])   # {"ad_spend": 1.84}
```

---

## How it works

```
Observations  (N samples, N vars)
    │
    ├─ CausalFFNN        pairwise sigmoid scorer → A ∈ ℝ^{N×N}
    │                    + NOTEARS acyclicity loss → no manual topo_order needed
    │
    ├─ from_embeddings   RMS embeddings E + exogenous residuals U = E − AᵀE
    │                    topo_order auto-extracted from A via Kahn's algorithm
    │
    ├─ NeuralSCM         COC type-checks the graph, rejects cycles as TypeError
    │
    ├─ do() / step()     subst(Var(X), Const(v)) + β-reduce → tensor value
    │                    propagation happens inside the COC kernel, not numpy
    │
    ├─ counterfactual()  Abduction → Action → Prediction
    │
    └─ CausalPlanner     Adam on Σ (‖E_next[j]‖ − target_j)²
```

**COC type guard** — the kernel runs before any matrix computation. A failed check leaves the SCM completely unmodified.

**Norm-based energy** — comparing embedding norms rather than full vectors eliminates direction-misalignment; the energy is exactly zero at the true solution.

**`do()` as β-reduction** — `do(X=v)` is `subst(mechanism, X, Const(v))` + `beta_reduce`. When both operands of `Add`/`Mul` are valued `Const`s, the kernel evaluates them directly: `App(App(Mul, Const(w)), Const(v)) → Const(w·v)`. The causal propagation happens inside the term language, not in a separate numpy pass.

**NOTEARS acyclicity** — during training, the augmented Lagrangian penalty `h(A) = tr(e^{A∘A}) − n` is added to the reconstruction loss. `h(A) = 0` iff `A` is a DAG. The multiplier schedule tightens automatically every 50 epochs.

---

## Simulating the world

CocDo's world model is a **Neural Structural Causal Model**: a directed graph where each node is a variable and each edge carries a learned continuous weight.

The world state lives in **embedding space** — each variable has a D-dimensional vector `E_j` whose L2 norm is its "intensity". The structural equation is:

```
E_j = Σᵢ A[i,j] · E_i  +  U_j
```

where `U_j = E_j − AᵀE_j` is the exogenous noise (what the parents don't explain).

**Simulating a trajectory** means choosing which variables to fix (`do()`), zeroing their incoming edges in `A`, and letting the rest propagate:

```python
# One causal step
state, E_next = scm.step({"rainfall": 2.0})

# Multi-step rollout — each step's E_next feeds the next
traj, reward = scm.rollout(
    action_sequence = [{"rainfall": 2.0}, {"rainfall": 1.5}, {}],
    reward_fn       = lambda s, t: s["crop_yield"],
    discount        = 0.95,
)
```

**Counterfactuals** answer "what would have happened if X had been different, given everything else we observed?" via Pearl's three-step: abduct `U` from observations, act by fixing X, predict by re-running the structural equations.

**Planning** closes the loop — instead of hand-picking intervention values, `CausalPlanner` searches for the `do()` values that minimise the distance to a target state, entirely by gradient descent through the causal graph.

The key modelling choice: **nodes are embeddings, not scalars**. This means any system whose state can be represented as vectors — token hidden states, sensor readings projected to a shared space, entity embeddings from a knowledge graph — can be dropped into CocDo without a separate observation model.

---

## Install

```bash
git clone https://github.com/lizixi-0x2F/CocDo
cd CocDo
pip install -e .
```

Requires Python ≥ 3.10, PyTorch ≥ 2.0, NumPy ≥ 1.24.

---

## Examples

### Synthetic DAG (gCastle)

```bash
pip install gcastle
python examples/demo_gcastle.py
```

Generates a random 5-node linear Gaussian DAG, trains `CausalFFNN` with NOTEARS constraint on 1000 observations, builds the SCM with auto-inferred topo_order, runs the planner:

```
CocDo learned continuous A (top edges):
  x0 → x1  (w=0.996)    x0 → x2  (w=0.940)    x0 → x3  (w=0.999)
  x1 → x3  (w=0.980)    x2 → x1  (w=0.997)    x2 → x3  (w=0.999)

Planner — recover x0 s.t. x3 = do(x0=3.0):
  optimal x0 = 3.0000   energy = 0.000000  ✓

Joint intervention (x0, x1, x2) — same target, multiple solutions:
  a_opt = {x0: 2.986, x1: 0.067, x2: 2.448}   energy = 0.000000  ✓
```

> **Note:** On Python ≥ 3.11, `pip install gcastle` works directly.

---

### CausalSearch — causal RAG over a knowledge base

```bash
pip install sentence-transformers
python examples/demo_causal_search.py
```

Indexes a markdown knowledge base (22 chapters, ~1800 paragraphs) and compares two retrieval strategies on the same queries:

**Vector search (RAG baseline):** cosine similarity in BGE embedding space.

**CausalSearch (Pearl three-step):**
1. **Abduction** — find the nearest-neighbor paragraph `j*` to the query in BGE space
2. **Action** — `do(j* = query_emb)`: inject the query vector, zero `j*`'s incoming edges in `A`
3. **Prediction** — `E_next = A_do^T @ E_do + U`; rank all nodes by `Δ‖E_next‖`

Positive Δ = **downstream activation** (knowledge triggered by the query).
Negative Δ = **upstream prerequisite** (concepts needed to understand the query).

```
Query: "What is the relationship between Transformer attention and Bayesian inference"

[Vector Search · RAG baseline]
   1. 'ch9·Transformer's success has triggered...'  cos=0.763
   2. 'ch9·Aside: attention as causality...'        cos=0.682
   ...

[CausalSearch · Pearl three-step]
  Abduction anchor -> 'ch9·Transformer's success...'  (cos=0.763)

  + Downstream (knowledge chain):
    + 'ch17·Comparing Bayesian update with ch14...'  delta=+2.69e-02
    + 'ch20·PAC and Bayes: ch17 left off...'         delta=+2.34e-02
    ...

  * CausalSearch-only (missed by RAG):
    ch17 Bayesian inference (x4), ch1 generative model layer, ch19 proof...
```

CausalSearch consistently surfaces cross-chapter prerequisite/consequence chains that cosine similarity misses — because it follows learned causal edges, not surface similarity.

---

### Tic-Tac-Toe — CausalPlanner vs MCTS

```bash
python examples/demo_tictactoe.py
```

Pits **gradient-based causal planning** against **Monte Carlo Tree Search** on Tic-Tac-Toe:

- **CausalPlanner** — treats move selection as a causal intervention problem. Each of the 9 cells is an SCM node with a 16-dim embedding; edges connect cells that share a winning line. The planner runs Adam gradient descent on the intervention energy — no tree search.
- **MCTS** — UCB1 selection + random rollouts (default 500). Near-optimal for Tic-Tac-Toe: always takes a winning move if available, always blocks a forced loss.

```bash
python examples/demo_tictactoe.py --games 50 --rollouts 500
```

```
══════════════════════════════════════════════════════════════
  Tic-Tac-Toe  |  CausalPlanner vs MCTS (500 rollouts/move)
══════════════════════════════════════════════════════════════

  Matchup A: Planner(X) vs MCTS(O)  — Planner moves first
    50/50  Planner=50  Draw=0  MCTS=0

  Matchup B: MCTS(X) vs Planner(O)  — MCTS moves first
    50/50  Planner=7   Draw=31  MCTS=12

  Overall  (100 games, MCTS=500 rollouts/move)
  CausalPlanner wins :  57 / 100  (57.0%)
  Draws              :  31 / 100  (31.0%)
  MCTS wins          :  12 / 100  (12.0%)
```

When moving first (X), CausalPlanner wins every game — the gradient signal from the SCM aligns cleanly with optimal play from the first move. When moving second (O), it draws or loses more often, as the board state requires multi-step lookahead the gradient planner can't model.

> Complexity: Planner = O(steps · N · D) per move (fast, differentiable) — MCTS = O(rollouts · depth) per move (exact, but discrete).

---

## API

### `CausalFFNN`

Learns the causal weight matrix from embeddings.

```python
ffnn = CausalFFNN(d_embed, hidden=256)
A, logits = ffnn(E)   # E: (B, N, D) or (N, D)  →  A: (N, N)
```

Each `A[i,j] ∈ (0,1)` is a sigmoid gate — causal strength of edge `i → j`. Diagonal forced to zero. Edges compete independently (sigmoid, not softmax), giving natural sparsity.

**Training with acyclicity constraint:**
```python
from cocdo.model.causal_ffnn import acyclicity_loss

rho, lam = 1.0, 0.0
for epoch in range(500):
    A, _ = ffnn(E_raw)
    recon = ((X @ A - X) ** 2).mean()
    h     = acyclicity_loss(A)          # = tr(e^{A∘A}) - n,  0 iff DAG
    loss  = recon + lam * h + (rho / 2) * h ** 2
    loss.backward(); optim.step()
    # tighten multipliers every N steps: lam += rho * h; rho *= 10 if needed
```

**Auto topo_order:**
```python
from cocdo.model.causal_ffnn import topo_order_from_A
order = topo_order_from_A(A_np, var_names)   # Kahn's algorithm on thresholded A
```

---

### `NeuralSCM`

```python
scm = NeuralSCM.from_embeddings(
    var_names,          # list of N node names
    A,                  # (N, N) causal weight matrix
    E_raw,              # (n_samples, N, D) per-sample embeddings
    topo_order=None,    # if None, auto-inferred from A
    edge_threshold=1e-4,
)
```

Internally: `E = sqrt(mean(E_raw²))` (RMS), `U = E − AᵀE` (exogenous residuals).

| Method | Description |
|--------|-------------|
| `do(var, value) → NeuralSCM` | Intervention — severs incoming edges, returns new SCM |
| `step(interventions, E_init)` | One step: `E_next = A_doᵀ E_do + U` |
| `rollout(action_sequence, reward_fn, discount)` | H-step rollout with discounted reward |
| `counterfactual(interventions, target)` | Pearl 3-step: abduction → action → prediction |

---

### `CausalPlanner`

```python
planner = CausalPlanner(scm)
result  = planner.plan(
    E_init,              # scm._E
    target,              # {var_name: desired_scalar_norm}
    interv_nodes,        # variables to optimise over
    lr=0.05,
    steps=200,
    rollout_steps=1,     # increase for multi-hop paths (X→Y→Z needs 2)
    cut_incoming=True,   # False when interv_nodes == target_nodes
)
# {"a_opt": {name: value}, "energy": float, "history": [...]}
```

Energy: `a* = argmin_a  Σⱼ (‖E_next[j]‖ − target_j)²`

`cut_incoming=True` (default) implements strict do-calculus — incoming edges of intervened nodes are severed. Set `cut_incoming=False` when the intervention nodes and target nodes overlap (e.g. optimising embeddings in-place), so gradient flows through the full `A`.

---

## COC kernel

```
cocdo/kernel/
├── terms.py      # AST: Sort, Var, Const, Pi, Lam, App, Add, Mul
├── reduction.py  # capture-avoiding substitution + tensor-aware β-reduction
└── typing.py     # type_of(), check_intervention()
```

- `Sort(i)` — universe level = topological depth. Root nodes are `Sort(0)`, their children `Sort(1)`, etc.
- `Pi(X, Typeᵢ, Typeⱼ)` — encodes edge `X → Y`. Requiring `i < j` makes cycles a type error.
- `subst(term, var, replacement)` — capture-avoiding; implements `do()`
- `beta_reduce(term)` — call-by-value to fixpoint; evaluates `Add`/`Mul` on valued `Const`s directly
- `Add`, `Mul` — built-in binary operators; `App(App(Mul, Const(w)), Const(v)) → Const(w·v)`

---

## Repository layout

```
cocdo/
├── kernel/
│   ├── terms.py        COC AST nodes + Add, Mul built-ins
│   ├── reduction.py    β-reduction + tensor-aware builtin evaluation
│   └── typing.py       type checker + intervention guard
└── model/
    ├── causal_ffnn.py  CausalFFNN, acyclicity_loss, topo_order_from_A
    ├── scm.py          NeuralSCM — do(), rollout(), counterfactual(), from_embeddings()
    └── planner.py      CausalPlanner — norm-based energy, Adam planning
examples/
├── demo_gcastle.py        gCastle synthetic DAG → full pipeline (NOTEARS + auto topo)
├── demo_causal_search.py  BGE + CausalFFNN → Pearl three-step causal RAG vs vector search
└── demo_tictactoe.py      CausalPlanner vs MCTS on Tic-Tac-Toe
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
