"""
gCastle → CocDo pipeline
========================
1. Generate synthetic data with ground-truth DAG via gCastle
2. Train CausalFFNN to learn continuous A matrix
3. Build NeuralSCM (continuous weights) + validate do() interventions
4. CausalPlanner: gradient-based optimal intervention search
"""
import sys, logging
import numpy as np
import torch

logging.basicConfig(level=logging.WARNING)

from castle.datasets import DAG, IIDSimulation

true_dag = DAG.erdos_renyi(n_nodes=5, n_edges=6, seed=42)
dataset  = IIDSimulation(W=true_dag, n=1000, method="linear", sem_type="gauss")
X        = dataset.X.astype(np.float32)   # (1000, 5)
B_true   = (true_dag != 0)               # (5, 5) ground-truth adjacency matrix
var_names = [f"x{i}" for i in range(5)]

print("=" * 60)
print("Ground-truth edges:")
for i in range(5):
    for j in range(5):
        if B_true[i, j]:
            print(f"  x{i} → x{j}  (w={true_dag[i,j]:.3f})")

# ── 2. Build token embeddings, train CausalFFNN ───────────────────────────────
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from cocdo import CausalFFNN, NeuralSCM
from cocdo.model import CausalPlanner

torch.set_default_dtype(torch.float32)

d_embed = 32
n_vars  = 5

np.random.seed(0)
proj  = (np.random.randn(n_vars, d_embed) / np.sqrt(d_embed)).astype(np.float32)
E_raw = torch.from_numpy(X[:, :, None] * proj[None, :, :]).float()  # (N, n_vars, d_embed)

ffnn  = CausalFFNN(d_embed=d_embed, hidden=128)
optim = torch.optim.Adam(ffnn.parameters(), lr=1e-3)

X_t = torch.from_numpy(X).float()   # (N, n_vars)

from cocdo.model.causal_ffnn import acyclicity_loss

print("\nTraining CausalFFNN (with NOTEARS acyclicity constraint) ...")
# Augmented Lagrangian schedule: start soft, tighten over training
rho    = 1.0    # quadratic penalty coefficient (grows)
lam    = 0.0    # Lagrange multiplier estimate (grows)
h_prev = float("inf")

for epoch in range(500):
    optim.zero_grad()
    A, _ = ffnn(E_raw)                          # (n_vars, n_vars)
    X_hat = X_t @ A                             # (N, n_vars)
    recon = ((X_hat - X_t) ** 2).mean()

    h = acyclicity_loss(A)                      # scalar ≥ 0, 0 iff DAG
    loss = recon + lam * h + (rho / 2) * h ** 2
    loss.backward()
    optim.step()

    # Update augmented Lagrangian multipliers every 50 epochs
    if (epoch + 1) % 50 == 0:
        with torch.no_grad():
            h_val = float(acyclicity_loss(ffnn(E_raw)[0]))
        lam = lam + rho * h_val
        if h_val > 0.25 * h_prev:
            rho = min(rho * 10, 1e6)
        h_prev = h_val
        print(f"  epoch {epoch+1:3d}  recon={recon.item():.4f}  h={h_val:.4f}  rho={rho:.1f}")

with torch.no_grad():
    A_final, _ = ffnn(E_raw)

A_np     = A_final.numpy()
E_raw_np = E_raw.numpy()

# topo_order now inferred automatically from A via Kahn's algorithm
scm = NeuralSCM.from_embeddings(
    var_names  = var_names,
    A          = A_np,
    E_raw      = E_raw_np,
    # topo_order omitted — auto-extracted from learned A
)

print("\nCocDo learned continuous A (top edges):")
for i in range(n_vars):
    for j in range(n_vars):
        if A_np[i, j] > 1e-3:
            print(f"  x{i} → x{j}  (w={A_np[i,j]:.4f})")

# ── 4. do() intervention ──────────────────────────────────────────────────────
print("\n== do() intervention ==")
baseline, _ = scm.step({})
print(f"  Baseline:   { {k: f'{v:.3f}' for k, v in baseline.items()} }")

intervened, _ = scm.step({"x0": 3.0})
print(f"  do(x0=3.0): { {k: f'{v:.3f}' for k, v in intervened.items()} }")

# The leaf node (highest topo level) is last in the auto-inferred topo_order
from cocdo.model.causal_ffnn import topo_order_from_A
_topo = topo_order_from_A(A_np, var_names)
target_var = _topo[-1]
cf = scm.counterfactual({"x0": 3.0}, target=target_var)
print(f"  Counterfactual E[{target_var} | do(x0=3.0)] = {cf:.4f}")

# ── 5. CausalPlanner gradient planning ────────────────────────────────────────
print("\n== CausalPlanner ==")
planner = CausalPlanner(scm)
assert scm._E is not None
E_plan = scm._E

target_scalar = intervened[target_var]
print(f"  Target: {target_var} = {target_scalar:.4f}")

result = planner.plan(
    E_init        = E_plan,
    target        = {target_var: target_scalar},
    interv_nodes  = ["x0"],
    lr            = 0.05,
    steps         = 500,
    rollout_steps = 1,
    verbose       = True,
)
print(f"  Optimal intervention (single): {result['a_opt']}")
print(f"  Final energy: {result['energy']:.6f}")
a_opt = result["a_opt"]["x0"]
state_opt, _ = scm.step({"x0": a_opt})
print(f"  Planned {target_var} = {state_opt[target_var]:.4f}  (target = {target_scalar:.4f})")
print(f"  Planner found x0 = {a_opt:.4f}  (true = 3.0000)")

# Joint intervention x0+x1+x2
print()
result3 = planner.plan(
    E_init        = scm._E,
    target        = {target_var: target_scalar},
    interv_nodes  = ["x0", "x1", "x2"],
    lr            = 0.05,
    steps         = 500,
    rollout_steps = 1,
    verbose       = True,
)
print(f"  Optimal intervention (joint): {result3['a_opt']}")
print(f"  Final energy: {result3['energy']:.6f}")
state3, _ = scm.step(result3["a_opt"])
print(f"  Planned {target_var} = {state3[target_var]:.4f}  (target = {target_scalar:.4f})")

print("\nDone.")
