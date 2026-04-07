"""
gCastle → CocDo 完整管线
========================
1. gCastle 生成带 ground-truth DAG 的合成数据
2. CausalFFNN 学习连续 A 矩阵
3. NeuralSCM 构建（连续权重）+ do() 干预验证
4. CausalPlanner 梯度规划求最优干预
"""
import sys, logging
import numpy as np
import torch

logging.basicConfig(level=logging.WARNING)

# ── 1. 合成数据（gCastle）──────────────────────────────────────────────────────
from castle.datasets import DAG, IIDSimulation

true_dag = DAG.erdos_renyi(n_nodes=5, n_edges=6, seed=42)
dataset  = IIDSimulation(W=true_dag, n=1000, method="linear", sem_type="gauss")
X        = dataset.X.astype(np.float32)   # (1000, 5)
B_true   = (true_dag != 0)               # (5, 5) ground-truth 邻接矩阵
var_names = [f"x{i}" for i in range(5)]

print("=" * 60)
print("ground-truth 因果边:")
for i in range(5):
    for j in range(5):
        if B_true[i, j]:
            print(f"  x{i} → x{j}  (w={true_dag[i,j]:.3f})")

# ── 2. 构造 token 嵌入，训练 CausalFFNN ───────────────────────────────────────
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from cocdo import CausalFFNN, NeuralSCM
from cocdo.model import CausalPlanner

torch.set_default_dtype(torch.float32)

d_embed = 32
n_vars  = 5

# 每个变量有独立随机投影方向，保证嵌入可区分
np.random.seed(0)
proj  = (np.random.randn(n_vars, d_embed) / np.sqrt(d_embed)).astype(np.float32)
E_raw = torch.from_numpy(X[:, :, None] * proj[None, :, :]).float()  # (N, n_vars, d_embed)

ffnn  = CausalFFNN(d_embed=d_embed, hidden=128)
optim = torch.optim.Adam(ffnn.parameters(), lr=1e-3)

# 标量观测张量，用于结构方程监督
X_t = torch.from_numpy(X).float()   # (N, n_vars)

print("\n训练 CausalFFNN ...")
for epoch in range(500):
    optim.zero_grad()
    A, _ = ffnn(E_raw)                          # (n_vars, n_vars)
    # 结构方程：x_hat[b,j] = Σ_i A[i,j] * x[b,i]
    X_hat = X_t @ A                             # (N, n_vars)
    loss  = ((X_hat - X_t) ** 2).mean()
    loss.backward()
    optim.step()

    if (epoch + 1) % 100 == 0:
        print(f"  epoch {epoch+1:3d}  loss={loss.item():.4f}  A_mean={A.mean().item():.4f}")

# ── 3. 构建 NeuralSCM（标准路径） ────────────────────────────────────────────
with torch.no_grad():
    A_final, _ = ffnn(E_raw)

A_np     = A_final.numpy()
E_raw_np = E_raw.numpy()

# 拓扑序：ground-truth 入度排序
in_degree  = B_true.astype(int).sum(axis=0)
topo_order = [var_names[i] for i in np.argsort(in_degree)]

scm = NeuralSCM.from_embeddings(
    var_names  = var_names,
    A          = A_np,
    E_raw      = E_raw_np,
    topo_order = topo_order,
)

print("\nCocDo 学到的连续 A 矩阵（所有正权重边）:")
for i in range(n_vars):
    for j in range(n_vars):
        if A_np[i, j] > 1e-3:
            print(f"  x{i} → x{j}  (w={A_np[i,j]:.4f})")

# ── 4. do() 干预验证 ──────────────────────────────────────────────────────────
print("\n== do() 干预验证 ==")
baseline, _ = scm.step({})
print(f"  基线: { {k: f'{v:.3f}' for k, v in baseline.items()} }")

intervened, _ = scm.step({"x0": 3.0})
print(f"  do(x0=3.0): { {k: f'{v:.3f}' for k, v in intervened.items()} }")

target_var = topo_order[-1]
cf = scm.counterfactual({"x0": 3.0}, target=target_var)
print(f"  反事实 E[{target_var} | do(x0=3.0)] = {cf:.4f}")

# ── 5. CausalPlanner 梯度规划 ─────────────────────────────────────────────────
print("\n== CausalPlanner 梯度规划 ==")
planner = CausalPlanner(scm)
assert scm._E is not None
E_plan = scm._E   # RMS 嵌入，由 from_embeddings 构建

# 目标：让 x3 达到 do(x0=3.0) 时的值（已知答案，验证 planner 能否反推 x0）
target_scalar = intervened[target_var]   # ~2.917，来自上面 do(x0=3.0) 的结果
print(f"  目标: {target_var} = {target_scalar:.4f}  (对应 do(x0=3.0) 的结果)")

result = planner.plan(
    E_init        = E_plan,
    target        = {target_var: target_scalar},
    interv_nodes  = ["x0"],
    lr            = 0.05,
    steps         = 500,
    rollout_steps = 1,
    verbose       = True,
)
print(f"  最优干预(单变量): {result['a_opt']}")
print(f"  最终能量: {result['energy']:.6f}")
a_opt = result["a_opt"]["x0"]
state_opt, _ = scm.step({"x0": a_opt})
print(f"  规划后 {target_var} = {state_opt[target_var]:.4f}  (目标 = {target_scalar:.4f})")
print(f"  planner 推出 x0 = {a_opt:.4f}  (真实 = 3.0000)")

# 联合干预 x0+x1+x2，验证多父节点时 planner 能完全收敛
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
print(f"  最优干预(三变量): {result3['a_opt']}")
print(f"  最终能量: {result3['energy']:.6f}")
state3, _ = scm.step(result3["a_opt"])
print(f"  规划后 {target_var} = {state3[target_var]:.4f}  (目标 = {target_scalar:.4f})")

print("\n管线完成。")
