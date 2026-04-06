# CocDo

**Causal discovery + do-calculus, grounded in type theory.**

CocDo learns a causal graph from observational data, then lets you run interventions (`do(X=v)`), counterfactuals, and multi-step causal rollouts — all type-checked by a lightweight COC kernel that rejects logically inconsistent queries at build time.

```python
from cocdo import CocDo

# 1. Train
model = CocDo(d_embed=32)
# ... training loop (see examples/) ...

# 2. Build causal graph
scm = model.build(pos_emb, var_names=["ad_spend", "clicks", "conversions"])

# 3. Intervene
state, _ = scm.step({"ad_spend": 3.0})
print(state["conversions"])   # causal effect, not correlation

# 4. Counterfactual
cf = model.counterfactual({"ad_spend": 3.0}, target="conversions")
```

## How it works

```
Observations (B, N)
    ↓  CausalFFNN          learns A ∈ R^{N×N}  (sigmoid edge weights)
    ↓  build()             COC type-checks the graph, rejects cycles
    ↓  do() / step()       Pearl do-calculus via term substitution + β-reduction
    ↓  counterfactual()    Abduction → Action → Prediction (Pearl Layer 3)
```

**COC type guard** — each causal edge `X → Y` is encoded as a dependent Pi-type `Π(X:Typeᵢ). Typeⱼ` where `i < j`. Any edge that would introduce a cycle raises `TypeError` before touching the neural state.

## Install

```bash
pip install -e .   # from repo root
```

Requires Python ≥ 3.10, PyTorch ≥ 2.0, NumPy ≥ 1.24.

## Quick demo

```bash
python examples/demo_ad_attribution.py
```

Expected output:

```
发现的因果图（child: [parents]）:
  ad_spend  (根节点)
  ad_spend → clicks
  clicks → conversions

== do() 干预：强制设定 ad_spend，看 conversions 怎么变 ==
  场景             ad_spend   clicks  conversions
  基线                0.507    0.812        0.561
  do(=1.0)          1.000    0.846        0.555
  do(=3.0)          3.000    1.861        0.537
```

The model discovers `ad_spend → clicks → conversions` from raw observations, with no graph structure provided.

## API

| Method | Description |
|---|---|
| `CocDo(d_embed, hidden)` | Create model |
| `model.train_step(y_batch, pos_emb, val_scale)` | Single training step, returns `{"loss": ...}` |
| `model.build(pos_emb, var_names, topo_order=None)` | Build `NeuralSCM` from learned weights |
| `model.do(var, value)` | Intervention — returns new `NeuralSCM` |
| `model.counterfactual(interventions, target)` | Pearl 3-step counterfactual |
| `scm.step(interventions)` | One-step causal propagation |
| `scm.rollout(action_sequence, reward_fn)` | H-step rollout with discounted reward |
| `model.causal_graph()` | `{child: [parents]}` dict |

## Use cases

- **Ad attribution** — isolate causal effect of spend from confounders
- **DeFi risk** — counterfactual stress-test: "if ETH drops 30%, what's liquidation volume?"
- **Root cause analysis** — which upstream variable actually caused the anomaly?
- **Mechanism design** — simulate policy changes before deploying them

## Architecture

```
cocdo/
├── kernel/
│   ├── terms.py       # COC AST: Sort, Var, Pi, Lam, App, Const
│   ├── reduction.py   # β-reduction + substitution
│   └── typing.py      # type_of(), check_intervention()
└── model/
    ├── causal_ffnn.py # Pairwise edge scorer → sigmoid A matrix
    ├── scm.py         # NeuralSCM: do(), counterfactual(), rollout()
    └── cocdo.py       # CocDo: train_step(), build(), unified interface
```

## Citation

If you use CocDo in research, please cite:

```bibtex
@software{cocdo2026,
  author = {lizixi},
  title  = {CocDo: Neural Causal Discovery with COC Type-Theoretic Interventions},
  year   = {2026},
  url    = {https://github.com/lizixi-0x2F/CocDo}
}
```
