"""
LLM Hidden State SCM Stack
===========================
从 SmolLM2-135M-Instruct 每层抽取 attention 权重和 hidden states，
直接用 attention 权重（多头 cat 后线性压缩）初始化 A，
为每层构建 NeuralSCM，串联成 30 层 SCM 堆栈。

架构：
  第 l 层 attention: (n_heads, N, N) → cat heads → (N, N*n_heads) → Linear → A_l (N, N)
  NeuralSCM_l: tokens 为节点，hidden states 为嵌入，A_l 为因果权重
  E_next = A_l^T @ E_do + U_l  →  NeuralSCM_{l+1}

用法：
  python examples/demo_llm_scm_stack.py
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from cocdo.model.causal_ffnn import acyclicity_loss, topo_order_from_A
from cocdo.model.scm import NeuralSCM

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEMO_TEXT = "The cat sat on the mat and looked at the dog."
TRAIN_EPOCHS = 300
LR = 1e-3


# ── 1. 抽取每层 hidden states + attention weights ────────────────────────────

def extract_llm_data(text: str):
    """
    返回:
      token_strs : N 个唯一 token 字符串
      hidden_layers : list[np.ndarray (N, D)]  — 30 层 hidden states
      attn_layers   : list[np.ndarray (n_heads, N, N)]  — 30 层 attention
    """
    print(f"加载模型 {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, attn_implementation="eager")
    model.eval()

    inputs = tokenizer(text, return_tensors="pt")
    token_ids = inputs["input_ids"][0].tolist()
    raw_strs = [tokenizer.decode([tid]) for tid in token_ids]

    seen: dict[str, int] = {}
    token_strs: list[str] = []
    for s in raw_strs:
        s = s.strip() or "<sp>"
        cnt = seen.get(s, 0)
        seen[s] = cnt + 1
        token_strs.append(f"{s}_{cnt}" if cnt > 0 else s)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

    hidden_layers = [
        outputs.hidden_states[l + 1][0].cpu().numpy()   # (N, D)
        for l in range(model.config.num_hidden_layers)
    ]
    attn_layers = [
        outputs.attentions[l][0].cpu().numpy()           # (n_heads, N, N)
        for l in range(model.config.num_hidden_layers)
    ]

    print(f"  tokens ({len(token_strs)}): {token_strs}")
    print(f"  hidden: {len(hidden_layers)} × {hidden_layers[0].shape}")
    print(f"  attn:   {len(attn_layers)} × {attn_layers[0].shape}")
    return token_strs, hidden_layers, attn_layers


# ── 2. 注意力蒸馏：用 attention 初始化 A，微调使其满足 NOTEARS ────────────────

class AttentionToA(nn.Module):
    """
    将 n_heads 个 attention 矩阵 concat 后线性压缩为单个 A (N, N)。

    每个 (i,j) 位置拼接所有头的值 → Linear(n_heads, 1) → sigmoid。
    保留全部头的信息，不做均值池化。
    """
    def __init__(self, n_heads: int):
        super().__init__()
        self.mix = nn.Linear(n_heads, 1, bias=True)

    def forward(self, attn: torch.Tensor) -> torch.Tensor:
        # attn: (n_heads, N, N) → (N, N, n_heads)
        _, N, _ = attn.shape
        x = attn.permute(1, 2, 0)                             # (N, N, n_heads)
        logits = self.mix(x).squeeze(-1)                      # (N, N)
        diag = torch.eye(N, device=attn.device, dtype=torch.bool)
        logits = logits.masked_fill(diag, float("-inf"))
        A = torch.sigmoid(logits).masked_fill(diag, 0.0)
        return A


def build_layer_scm(
    layer_idx: int,
    hidden: np.ndarray,        # (N, D)
    attn: np.ndarray,          # (n_heads, N, N)
    var_names: list[str],
    epochs: int = TRAIN_EPOCHS,
    lr: float = LR,
    verbose: bool = False,
) -> NeuralSCM:
    """
    用注意力蒸馏训练 A，返回对应的 NeuralSCM。

    损失 = 蒸馏损失 + NOTEARS 无环惩罚
      distill = ||A - attn_mean||_F^2   (A 拟合 attention 均值)
      h(A)    = tr(e^{A∘A}) - n         (NOTEARS 精确约束)
      loss    = distill + λ·h + (ρ/2)·h^2
    """
    n_heads, N, _ = attn.shape
    attn_t = torch.tensor(attn, dtype=torch.float32)          # (n_heads, N, N)
    attn_mean = torch.tensor(attn.mean(0), dtype=torch.float32)  # (N, N) 目标

    net = AttentionToA(n_heads=n_heads)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    rho, lam = 1.0, 0.0
    for ep in range(epochs):
        opt.zero_grad()
        A = net(attn_t)                                        # (N, N)
        distill = ((A - attn_mean) ** 2).mean()
        h = acyclicity_loss(A)
        loss = distill + lam * h + (rho / 2) * h * h
        loss.backward()
        opt.step()

        if (ep + 1) % 100 == 0:
            h_val = h.item()
            lam += rho * h_val
            if h_val > 0.25:
                rho *= 5.0
            if verbose:
                print(f"  layer {layer_idx:2d} ep {ep+1:3d}  "
                      f"distill={distill.item():.4f}  h={h_val:.4f}")

    with torch.no_grad():
        A_np = net(attn_t).numpy()                             # (N, N)

    topo = topo_order_from_A(A_np, var_names)
    E_raw = hidden[np.newaxis]                                 # (1, N, D)
    return NeuralSCM.from_embeddings(
        var_names=var_names,
        A=A_np,
        E_raw=E_raw,
        topo_order=topo,
    )


# ── 3. SCMStack ───────────────────────────────────────────────────────────────

class SCMStack:
    def __init__(self, scms: list[NeuralSCM], var_names: list[str]):
        self.scms = scms
        self.var_names = var_names
        self.n_layers = len(scms)

    def propagate(
        self,
        interventions: dict[str, float],
        start_layer: int = 0,
    ) -> list[dict[str, float]]:
        trajectory: list[dict[str, float]] = []
        E_cur = None
        for l in range(start_layer, self.n_layers):
            scm = self.scms[l]
            state, E_cur = scm.step(interventions, E_init=E_cur)
            trajectory.append(state)
            interventions = {}
            # 层间 LN：把 E_cur rescale 到当前层 _E 的 mean norm
            target = float(np.linalg.norm(scm._E, axis=-1).mean())
            cur = np.linalg.norm(E_cur, axis=-1, keepdims=True).clip(min=1e-8)
            E_cur = E_cur / cur * target
        return trajectory

    def layer_edges(self, layer_idx: int) -> list[tuple[str, str, float]]:
        scm = self.scms[layer_idx]
        edges = []
        for name, node in scm.vars.items():
            for p in node.parents:
                edges.append((p, name, node.parent_weights[p]))
        return sorted(edges, key=lambda x: -x[2])

    def summary(self):
        print(f"\nSCMStack: {self.n_layers} 层, {len(self.var_names)} 个 token 节点")
        for l, scm in enumerate(self.scms):
            n_edges = sum(len(n.parents) for n in scm.vars.values())
            print(f"  层 {l:2d}: {n_edges:3d} 条因果边")


# ── 4. 主流程 ─────────────────────────────────────────────────────────────────

def main():
    var_names, hidden_layers, attn_layers = extract_llm_data(DEMO_TEXT)

    print(f"\n为 {len(hidden_layers)} 层各训练 AttentionToA ({TRAIN_EPOCHS} epochs) ...")
    scms: list[NeuralSCM] = []
    for l, (hidden, attn) in enumerate(zip(hidden_layers, attn_layers)):
        verbose = (l % 10 == 9)
        scm = build_layer_scm(l, hidden, attn, var_names, verbose=verbose)
        scms.append(scm)
        if (l + 1) % 5 == 0:
            n_edges = sum(len(n.parents) for n in scm.vars.values())
            print(f"  层 {l+1:2d}/30 完成，边数={n_edges}")

    stack = SCMStack(scms, var_names)
    stack.summary()

    print("\n── 各层 Top-3 因果边 ──")
    for l in [0, 9, 19, 29]:
        edges = stack.layer_edges(l)[:3]
        print(f"  层 {l:2d}: " + "  |  ".join(f"{p}→{c} ({w:.3f})" for p, c, w in edges))

    first_tok = var_names[0]
    last_tok  = var_names[-1]
    print(f"\n── 干预实验: do({first_tok!r} = 2.0) ──")
    baseline   = stack.propagate({})
    intervened = stack.propagate({first_tok: 2.0})
    print(f"  {'层':>4}  {'基线 '+last_tok:>18}  {'干预 '+last_tok:>18}  {'差值':>10}")
    for l in range(stack.n_layers):
        b = baseline[l].get(last_tok, 0.0)
        v = intervened[l].get(last_tok, 0.0)
        print(f"  {l:4d}  {b:18.4f}  {v:18.4f}  {v-b:+10.4f}")

    print("\n完成。")


if __name__ == "__main__":
    main()
