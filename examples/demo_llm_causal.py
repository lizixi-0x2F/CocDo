"""
LLM Attention → CausalSCM
==========================
Extract all attention maps from SmolLM2-135M-Instruct, pool them with
AdaptiveAvgPool2d across (layers, heads) into a single (T, T) causal
weight matrix A_attn, then feed it into NeuralSCM for do() interventions
and gradient planning over token representations.
"""
import sys
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from cocdo import NeuralSCM
from cocdo.model import CausalPlanner

# ── 1. Load model ──────────────────────────────────────────────────────────────
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
print("Loading SmolLM2-135M-Instruct ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float32,
    attn_implementation="eager",   # required for output_attentions=True
)
model.eval()

# ── 2. Forward pass — collect all attention maps ───────────────────────────────
PROMPT = "The cause of the French Revolution was economic inequality."
inputs = tokenizer(PROMPT, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
T = len(tokens)
print(f"\nPrompt : {PROMPT!r}")
print(f"Tokens ({T}): {tokens}")

with torch.no_grad():
    outputs = model(
        **inputs,
        output_attentions=True,
        output_hidden_states=True,
    )

# outputs.attentions: tuple of L tensors, each (1, H, T, T)
# outputs.hidden_states: tuple of L+1 tensors, each (1, T, D)
L = len(outputs.attentions)   # 30
H = outputs.attentions[0].shape[1]   # 9

print(f"\nLayers={L}  Heads={H}  SeqLen={T}")

# ── 3. AdaptivePool across (layers, heads) → A_attn (T, T) ────────────────────
#
# Stack all attention maps: (L, H, T, T)
# Reshape to treat each (i,j) position as a spatial location: (T*T, L, H)
# AdaptiveAvgPool2d(1, 1): pool L and H down to (1, 1) per position
# Reshape back: (T, T)

attn_stack = torch.stack([
    outputs.attentions[l][0]   # (H, T, T), drop batch dim
    for l in range(L)
])  # (L, H, T, T)

# (L, H, T, T) → (T*T, L, H) so AdaptiveAvgPool2d sees (batch=T*T, C=L, H, W=H)
pool = nn.AdaptiveAvgPool2d((1, 1))
attn_flat = attn_stack.permute(2, 3, 0, 1).reshape(T * T, L, H)  # (T*T, L, H)
# treat as (T*T, 1, L, H) for the 2D pool
A_attn = pool(attn_flat.unsqueeze(1)).squeeze()  # (T*T,)
A_attn = A_attn.reshape(T, T).numpy()            # (T, T)

# A_attn[i,j] = "token at position j attends to token at position i"
# NeuralSCM.step() computes E_next = A^T @ E, i.e. uses columns of A.
# We want E_next[j] = sum_i A_attn[i,j] * E[i], which is already A^T @ E
# with A = A_attn. But topo_order requires A[i,j]>0 only when i<j (parent→child).
# A_attn rows are the attention sources, so we transpose to get the right shape:
#   A_scm[i,j] = A_attn[j,i] = "how much j attends to i" → i is parent of j
A_scm = A_attn.T   # (T, T): A_scm[i,j] = weight of edge i→j

# ── 4. Build NeuralSCM ─────────────────────────────────────────────────────────
# Use last-layer hidden states only — avoids norm explosion from stacking layers.
# Shape: (1, T, D) as E_raw (n_samples=1, N=T, D=576)
hidden_states = outputs.hidden_states[-1][0].unsqueeze(0).numpy()  # (1, T, D)

# Token position order = topological order (causal LM is already a DAG)
topo_order = [f"t{i}" for i in range(T)]
var_names  = topo_order

scm = NeuralSCM.from_embeddings(
    var_names  = var_names,
    A          = A_scm,
    E_raw      = hidden_states,
    topo_order = topo_order,
)
# Zero out U so causal propagation is fully driven by A_attn.
# With attention weights (row-sum=1), U≈E dominates and masks interventions.
import numpy as np
scm._U = np.zeros_like(scm._U)

print("\nTop causal edges (A_scm[i,j] > 0.05, i→j):")
for i in range(T):
    for j in range(T):
        if i != j and A_scm[i, j] > 0.05:
            print(f"  {tokens[i]:12s} → {tokens[j]:12s}  (w={A_scm[i,j]:.4f})")

# ── 5. do() intervention ──────────────────────────────────────────────────────
# Intervene on "economic" (t7) and observe "inequality" (t8) and "." (t9)
interv_tok = f"t{T-3}"   # "economic"
target_tok = f"t{T-2}"   # "inequality"

print(f"\n== do() intervention on '{tokens[T-3]}' ==")
baseline, _ = scm.step({})
print("  Baseline norms:", {tokens[int(k[1:])]: f"{v:.3f}" for k, v in baseline.items()})

intervened, _ = scm.step({interv_tok: baseline[interv_tok] * 3.0})
print(f"  do({tokens[T-3]}=3x): " + str({tokens[int(k[1:])]: f"{v:.3f}" for k, v in intervened.items()}))

# ── 6. Counterfactual ──────────────────────────────────────────────────────────
cf = scm.counterfactual({interv_tok: baseline[interv_tok] * 3.0}, target=target_tok)
base_norm = baseline[target_tok]
print(f"\n  Counterfactual: '{tokens[T-2]}' norm")
print(f"    baseline              = {base_norm:.4f}")
print(f"    do({tokens[T-3]}=3x) = {cf:.4f}  (Δ = {cf - base_norm:+.4f}  {(cf/base_norm-1)*100:+.1f}%)")

# ── 7. CausalPlanner ──────────────────────────────────────────────────────────
print(f"\n== CausalPlanner: push '{tokens[T-2]}' to 1.5x baseline ==")
planner     = CausalPlanner(scm)
target_norm = base_norm * 1.5
assert scm._E is not None

# Single intervention on "economic"
result = planner.plan(
    E_init       = scm._E,
    target       = {target_tok: target_norm},
    interv_nodes = [interv_tok],
    lr           = 0.1,
    steps        = 300,
    rollout_steps= T,
    verbose      = True,
)
print(f"  target norm                = {target_norm:.4f}")
print(f"  optimal {tokens[T-3]:10s}      = {result['a_opt'][interv_tok]:.4f}  (baseline = {base_norm:.4f})")
print(f"  final energy               = {result['energy']:.6f}")
state_opt, _ = scm.step({interv_tok: result["a_opt"][interv_tok]})
print(f"  achieved '{tokens[T-2]}' norm = {state_opt[target_tok]:.4f}")

# Joint intervention: "The" + "economic" → "inequality"
print(f"\n  Joint intervention: 'The' + '{tokens[T-3]}' → '{tokens[T-2]}'")
result_joint = planner.plan(
    E_init       = scm._E,
    target       = {target_tok: target_norm},
    interv_nodes = ["t0", interv_tok],
    lr           = 0.1,
    steps        = 300,
    rollout_steps= T,
    verbose      = True,
)
print(f"  optimal: {{'The': {result_joint['a_opt']['t0']:.4f}, '{tokens[T-3]}': {result_joint['a_opt'][interv_tok]:.4f}}}")
print(f"  final energy = {result_joint['energy']:.6f}")
state_j, _ = scm.step(result_joint["a_opt"])
print(f"  achieved '{tokens[T-2]}' norm = {state_j[target_tok]:.4f}  (target = {target_norm:.4f})")

print("\nDone.")
