"""
CocDo — Ad Attribution Demo
============================
场景：广告投放 → 点击率 → 转化率

真实因果链：
    ad_spend → clicks → conversions

我们用合成数据训练模型，让它自己发现这条因果链，
然后用 do() 做干预：
    "如果我把广告预算强制拉高，转化会变多少？"

运行：
    python examples/demo_ad_attribution.py
"""
import torch
import torch.optim as optim
import numpy as np

from cocdo import CocDo

# ── 0. 可复现 ────────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── 1. 合成数据生成（真实因果链：ad_spend → clicks → conversions） ────────────
N_VARS   = 3
VAR_NAMES = ["ad_spend", "clicks", "conversions"]
B        = 512   # 样本数

# 真实生成过程
ad_spend    = np.random.uniform(0.0, 10.0, B)          # 广告投放，独立变量
clicks      = 0.6 * ad_spend + np.random.randn(B) * 0.5
conversions = 0.4 * clicks   + np.random.randn(B) * 0.3

# 归一化到 [-1, 1]
def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-8)

data = np.stack([normalize(ad_spend), normalize(clicks), normalize(conversions)], axis=1)
y_batch = torch.tensor(data, dtype=torch.float32)   # (B, N)

print(f"数据形状: {y_batch.shape}  (样本数={B}, 变量数={N_VARS})")

# ── 2. 初始化模型 + 可学习参数 ────────────────────────────────────────────────
D_EMBED = 32
model     = CocDo(d_embed=D_EMBED, hidden=128)
pos_emb   = torch.nn.Parameter(torch.randn(N_VARS, D_EMBED) * 0.1)
val_scale = torch.nn.Parameter(torch.randn(D_EMBED) * 0.1)

optimizer = optim.Adam(
    list(model.parameters()) + [pos_emb, val_scale],
    lr=1e-3,
)

# ── 3. 训练 ───────────────────────────────────────────────────────────────────
EPOCHS = 300
print("\n训练中...")
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    out  = model.train_step(y_batch, pos_emb, val_scale)
    loss = out["loss"]
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"  epoch {epoch+1:3d}/{EPOCHS}  loss={loss.item():.4f}")

# ── 4. 构建因果图（build） ────────────────────────────────────────────────────
# topo_order 告诉模型已知的拓扑顺序，防止逆向边通过类型检查
scm = model.build(
    pos_emb   = pos_emb.detach(),
    var_names = VAR_NAMES,
    val_scale = val_scale.detach(),
    topo_order = VAR_NAMES,   # ad_spend < clicks < conversions
)

print("\n发现的因果图（child: [parents]）:")
graph = model.causal_graph()
for node, parents in graph.items():
    if parents:
        print(f"  {' + '.join(parents)} → {node}")
    else:
        print(f"  {node}  (根节点)")

# ── 5. do() 干预 ──────────────────────────────────────────────────────────────
print("\n== do() 干预：强制设定 ad_spend，看 conversions 怎么变 ==")

baseline_state, _  = scm.step({})                       # 无干预基线
do_low,  _         = scm.step({"ad_spend": 1.0})        # 小预算
do_high, _         = scm.step({"ad_spend": 3.0})        # 大预算

print(f"  {'场景':<12} {'ad_spend':>10} {'clicks':>8} {'conversions':>12}")
print(f"  {'基线':<12} {baseline_state['ad_spend']:>10.3f} {baseline_state['clicks']:>8.3f} {baseline_state['conversions']:>12.3f}")
print(f"  {'do(=1.0)':<12} {do_low['ad_spend']:>10.3f} {do_low['clicks']:>8.3f} {do_low['conversions']:>12.3f}")
print(f"  {'do(=3.0)':<12} {do_high['ad_spend']:>10.3f} {do_high['clicks']:>8.3f} {do_high['conversions']:>12.3f}")
print(f"\n  加大预算的因果效应（conversions）: {do_high['conversions'] - do_low['conversions']:+.4f}")

# ── 6. 反事实 ─────────────────────────────────────────────────────────────────
print("\n== 反事实：如果广告预算是 3.0 而不是当前值，转化会是多少？ ==")
cf_value = model.counterfactual(
    interventions={"ad_spend": 3.0},
    target="conversions",
)
print(f"  反事实 conversions: {cf_value:.4f}")

# ── 7. 多步 rollout ───────────────────────────────────────────────────────────
print("\n== 3步 rollout：逐步加大广告投放 ==")
action_sequence = [
    {"ad_spend": 1.0},
    {"ad_spend": 2.0},
    {"ad_spend": 3.0},
]

def reward_fn(state, _t):
    return state["conversions"]

trajectory, total_reward = scm.rollout(action_sequence, reward_fn)
print(f"  {'step':<6} {'ad_spend':>10} {'clicks':>8} {'conversions':>12}")
for t, state in enumerate(trajectory):
    print(f"  {t:<6} {state['ad_spend']:>10.3f} {state['clicks']:>8.3f} {state['conversions']:>12.3f}")
print(f"  累计 reward: {total_reward:.4f}")

print("\n完成。")
