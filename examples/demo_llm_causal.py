"""
LLM Causal Intervention → Generation Steering
==============================================
Build causal SCM from attention maps, then steer generation by
intervening on token representations before the model continues writing.
"""
import sys
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from cocdo import NeuralSCM
from cocdo.model import CausalPlanner

# ── Load ───────────────────────────────────────────────────────────────────────
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
print("Loading SmolLM2-135M-Instruct ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float32, attn_implementation="eager"
)
model.eval()

# ── Forward pass — collect attentions + hidden states ─────────────────────────
PROMPT  = "The cause of the French Revolution was economic inequality"
inputs  = tokenizer(PROMPT, return_tensors="pt")
tokens  = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
T       = len(tokens)
print(f"\nPrompt: {PROMPT!r}  ({T} tokens)")

with torch.no_grad():
    outputs = model(**inputs, output_attentions=True, output_hidden_states=True)

L = len(outputs.attentions)
H = outputs.attentions[0].shape[1]

# ── AdaptiveAvgPool (L, H) → A_attn (T, T) ────────────────────────────────────
attn_stack = torch.stack([outputs.attentions[l][0] for l in range(L)])  # (L,H,T,T)
pool       = nn.AdaptiveAvgPool2d((1, 1))
attn_flat  = attn_stack.permute(2, 3, 0, 1).reshape(T * T, L, H)
A_attn     = pool(attn_flat.unsqueeze(1)).squeeze().reshape(T, T).numpy()
A_scm      = A_attn.T   # A_scm[i,j] = weight i→j

# ── Build NeuralSCM ───────────────────────────────────────────────────────────
E_raw_np   = outputs.hidden_states[-1][0].unsqueeze(0).numpy()  # (1, T, D)
topo_order = [f"t{i}" for i in range(T)]
scm        = NeuralSCM.from_embeddings(
    var_names=topo_order, A=A_scm, E_raw=E_raw_np, topo_order=topo_order
)
# U from self-attention diagonal
assert scm._E is not None
scm._U = np.diag(A_attn)[:, None] * scm._E

# ── Generation helper ─────────────────────────────────────────────────────────
def generate_from_hidden(
    h_last: np.ndarray,
    input_ids: torch.Tensor,
    max_new: int = 30,
) -> str:
    """
    Replace the last token's hidden state with h_last, then greedily generate.
    We splice the intervened hidden into the model by using it as the KV cache
    seed: run one forward pass with the modified last hidden → get next token →
    append → repeat normally.
    """
    h_t   = torch.from_numpy(h_last).float().unsqueeze(0).unsqueeze(0)  # (1,1,D)
    # Project modified hidden → logits → sample next token
    logits_first = model.lm_head(h_t).squeeze(0)   # (1, vocab)
    next_id      = logits_first.argmax(dim=-1)       # (1,)
    generated    = [next_id.item()]

    # Continue generation normally from the full context + first new token
    cur_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=-1)
    for _ in range(max_new - 1):
        with torch.no_grad():
            out     = model(cur_ids)
            next_id = out.logits[0, -1].argmax().unsqueeze(0)
        generated.append(next_id.item())
        cur_ids = torch.cat([cur_ids, next_id.unsqueeze(0)], dim=-1)
        if next_id.item() in (tokenizer.eos_token_id, tokenizer.pad_token_id):
            break

    return tokenizer.decode(generated, skip_special_tokens=True)

# ── Baseline generation ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("BASELINE (no intervention)")
print("=" * 65)
_, E_base = scm.step({})
continuation_base = generate_from_hidden(E_base[T-1], inputs["input_ids"])
print(f"  {PROMPT} ...")
print(f"  → {continuation_base}")

# ── Intervene on different tokens, watch generation change ────────────────────
interventions = {
    "economic (t7)":    (f"t{T-2}", [2.0, 5.0, 10.0]),
    "Revolution (t5)":  (f"t5",     [2.0, 5.0, 10.0]),
    "The (t0)":         (f"t0",     [2.0, 5.0, 10.0]),
}

for label, (node, scales) in interventions.items():
    tok_idx  = int(node[1:])
    tok_name = tokens[tok_idx]
    base_val = float(np.linalg.norm(scm._E[tok_idx]))
    print(f"\n{'=' * 65}")
    print(f"INTERVENING ON  '{tok_name}'  (baseline norm = {base_val:.2f})")
    print(f"{'=' * 65}")
    for scale in scales:
        interv_val       = base_val * scale
        _, E_int         = scm.step({node: interv_val})
        continuation_int = generate_from_hidden(E_int[T-1], inputs["input_ids"])
        print(f"  do({tok_name}={scale:.0f}x) → {continuation_int}")

# ── CausalPlanner: find intervention to steer toward a target concept ──────────
print(f"\n{'=' * 65}")
print("CAUSALPLANNER: push 'inequality' norm toward 'Revolution' level")
print(f"{'=' * 65}")

# Target: make "inequality" have same representation magnitude as "Revolution"
rev_norm  = float(np.linalg.norm(scm._E[5]))   # Revolution = t5
ineq_node = f"t{T-1}"
econ_node = f"t{T-2}"

planner = CausalPlanner(scm)
result  = planner.plan(
    E_init       = scm._E,
    target       = {ineq_node: rev_norm},
    interv_nodes = [econ_node, "t0"],
    lr           = 0.05,
    steps        = 500,
    rollout_steps= T,
    verbose      = True,
)
a_opt = result["a_opt"]
print(f"  target norm ({tokens[T-1]}) = {rev_norm:.4f}")
print(f"  optimal interventions: { {tokens[int(k[1:])]: f'{v:.3f}' for k, v in a_opt.items()} }")
print(f"  final energy = {result['energy']:.6f}")

_, E_planned        = scm.step(a_opt)
continuation_plan   = generate_from_hidden(E_planned[T-1], inputs["input_ids"])
print(f"\n  Baseline  → {continuation_base}")
print(f"  Planned   → {continuation_plan}")

print("\nDone.")
