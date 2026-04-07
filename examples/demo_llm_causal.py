"""
LLM Attention → CausalSCM with logit-level interventions
=========================================================
1. Extract all attention maps from SmolLM2-135M-Instruct
2. AdaptiveAvgPool2d across (layers, heads) → A_attn (T, T)
3. Model U from diagonal (self-attention residual)
4. Build NeuralSCM over token representations
5. do() interventions → observe next-token logit shifts
6. CausalPlanner → find optimal token intervention to steer predictions
"""
import sys
import numpy as np
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
    MODEL, dtype=torch.float32, attn_implementation="eager"
)
model.eval()

lm_head  = model.lm_head          # (D, vocab) — projects hidden → logits
# SmolLM2 ties embeddings, so lm_head.weight is (vocab, D)

def hidden_to_top_tokens(h: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
    """Project a (D,) hidden state to top-k predicted tokens + probabilities."""
    with torch.no_grad():
        logits = lm_head(torch.from_numpy(h).float().unsqueeze(0)).squeeze(0)
        probs  = torch.softmax(logits, dim=-1)
        topk   = torch.topk(probs, k)
    return [
        (tokenizer.decode([idx.item()]), float(p))
        for idx, p in zip(topk.indices, topk.values)
    ]

# ── 2. Forward pass ────────────────────────────────────────────────────────────
PROMPT = "The cause of the French Revolution was economic inequality"
inputs = tokenizer(PROMPT, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
T = len(tokens)
print(f"\nPrompt : {PROMPT!r}")
print(f"Tokens ({T}): {tokens}")

with torch.no_grad():
    outputs = model(**inputs, output_attentions=True, output_hidden_states=True)

L = len(outputs.attentions)    # 30
H = outputs.attentions[0].shape[1]   # 9
print(f"Layers={L}  Heads={H}  SeqLen={T}")

# ── 3. AdaptiveAvgPool across (L, H) → A_attn (T, T) ─────────────────────────
attn_stack = torch.stack([outputs.attentions[l][0] for l in range(L)])  # (L,H,T,T)

pool      = nn.AdaptiveAvgPool2d((1, 1))
attn_flat = attn_stack.permute(2, 3, 0, 1).reshape(T * T, L, H)
A_attn    = pool(attn_flat.unsqueeze(1)).squeeze().reshape(T, T).numpy()  # (T,T)

# Transpose: A_scm[i,j] = "how much j attends to i" → i is parent of j
A_scm = A_attn.T   # (T,T): A_scm[i,j] = weight of edge i→j

print(f"\nTop causal edges (A_scm[i,j] > 0.05):")
for i in range(T):
    for j in range(T):
        if i != j and A_scm[i, j] > 0.05:
            print(f"  {tokens[i]:14s} → {tokens[j]:14s}  (w={A_scm[i,j]:.4f})")

# ── 4. Build NeuralSCM with proper U ──────────────────────────────────────────
# E_raw: last-layer hidden states, shape (1, T, D)
E_raw_t      = outputs.hidden_states[-1][0]              # (T, D)
hidden_np    = E_raw_t.unsqueeze(0).numpy()              # (1, T, D)

topo_order = [f"t{i}" for i in range(T)]
var_names  = topo_order

scm = NeuralSCM.from_embeddings(
    var_names  = var_names,
    A          = A_scm,
    E_raw      = hidden_np,
    topo_order = topo_order,
)

# U[j] = self-attention residual = diagonal weight * E[j]
# A_attn[j,j] is how much token j attends to itself (pooled across L,H)
assert scm._E is not None
diag_weights = np.diag(A_attn)                     # (T,) self-attention weight
scm._U = diag_weights[:, None] * scm._E            # (T, D)

print(f"\nSelf-attention diagonal (U scale): {np.round(diag_weights, 3)}")

# ── 5. Baseline next-token prediction from last hidden state ──────────────────
print("\n== Baseline next-token predictions (from last hidden state) ==")
E_last_baseline = scm._E[-1]   # (D,) last token hidden state
baseline_top = hidden_to_top_tokens(E_last_baseline)
print(f"  Last token: '{tokens[-1]}'")
print(f"  Top-5 next tokens: {baseline_top}")

# ── 6. do() intervention → logit shift ────────────────────────────────────────
# Intervene on "economic" (T-2), propagate, observe last-token predictions
interv_node = f"t{T-2}"   # "economic" (second-to-last)
target_node = f"t{T-1}"   # last token = "inequality"

print(f"\n== do('{tokens[T-2]}') interventions → next-token prediction shift ==")

baseline_state, E_base_next = scm.step({})
E_last_base = E_base_next[T-1]
base_preds  = hidden_to_top_tokens(E_last_base)
print(f"  Baseline  → top-5: {base_preds}")

for scale in [2.0, 5.0, 10.0]:
    interv_val = float(np.linalg.norm(scm._E[T-2])) * scale
    state_int, E_int_next = scm.step({interv_node: interv_val})
    E_last_int = E_int_next[T-1]
    int_preds  = hidden_to_top_tokens(E_last_int)
    delta_norm = state_int[target_node] - baseline_state[target_node]
    print(f"  do({tokens[T-2]}={scale:.0f}x) → top-5: {int_preds}  Δnorm={delta_norm:+.3f}")

# ── 7. Counterfactual on last token ───────────────────────────────────────────
print(f"\n== Counterfactual: '{tokens[-1]}' hidden state ==")
cf_val  = scm.counterfactual({interv_node: float(np.linalg.norm(scm._E[T-2])) * 5.0},
                              target=target_node)
base_norm = float(np.linalg.norm(scm._E[T-1]))
print(f"  baseline norm   = {base_norm:.4f}")
print(f"  do(economic=5x) = {cf_val:.4f}  (Δ = {cf_val - base_norm:+.4f}  {(cf_val/base_norm-1)*100:+.1f}%)")

# ── 8. CausalPlanner: steer last-token hidden toward a target embedding ────────
# Target: the hidden state of "economic" (make "inequality" look like "economic")
print(f"\n== CausalPlanner: steer '{tokens[-1]}' representation ==")

# Target norm = norm of "economic"'s own hidden state
economic_norm  = float(np.linalg.norm(scm._E[T-2]))
target_norm    = economic_norm   # push "inequality" to have same magnitude as "economic"

planner = CausalPlanner(scm)
result  = planner.plan(
    E_init       = scm._E,
    target       = {target_node: target_norm},
    interv_nodes = [interv_node],
    lr            = 0.05,
    steps         = 500,
    rollout_steps = T,
    verbose       = True,
)
a_opt = result["a_opt"][interv_node]
print(f"  target norm ({tokens[-1]}) = {target_norm:.4f}")
print(f"  optimal {tokens[T-2]:12s} = {a_opt:.4f}  (baseline = {economic_norm:.4f})")
print(f"  final energy           = {result['energy']:.6f}")

state_opt, E_opt_next = scm.step({interv_node: a_opt})
E_last_opt  = E_opt_next[T-1]
opt_preds   = hidden_to_top_tokens(E_last_opt)
print(f"  achieved '{tokens[-1]}' norm = {state_opt[target_node]:.4f}")
print(f"  next-token predictions after planning: {opt_preds}")

print(f"\n  Prediction shift summary:")
print(f"    baseline → {base_preds[0]}")
print(f"    planned  → {opt_preds[0]}")

print("\nDone.")
