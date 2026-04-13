"""
Demo of CausalNetwork — end-to-end causal reasoning neural network.

This demo shows:
1. Training the causal network from observational data
2. Predicting intervention effects
3. Counterfactual reasoning
4. Gradient-based causal planning

All through a unified neural network interface.
"""

import sys
import logging
import numpy as np
import torch

logging.basicConfig(level=logging.INFO)

# Add parent directory to path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from cocdo.model import CausalNetwork
from cocdo.model.causal_ffnn import acyclicity_loss


# ── Generate synthetic data ──────────────────────────────────────────────────
def generate_linear_gaussian_dag(n_vars=5, n_samples=1000, seed=42):
    """Generate linear Gaussian data from a random DAG."""
    np.random.seed(seed)

    # Random DAG adjacency matrix
    B = np.random.rand(n_vars, n_vars) * 0.5
    B = np.triu(B, k=1)  # Upper triangular = no cycles
    B[np.random.rand(*B.shape) < 0.7] = 0  # Add sparsity

    # Generate data: X = B^T @ X + noise
    X = np.zeros((n_samples, n_vars))
    for i in range(n_vars):
        parents = np.where(B[:, i] != 0)[0]
        if len(parents) > 0:
            X[:, i] = X[:, parents] @ B[parents, i] + np.random.randn(n_samples) * 0.1
        else:
            X[:, i] = np.random.randn(n_samples) * 0.5

    # Add embedding dimension (D=32)
    d_embed = 32
    proj = np.random.randn(n_vars, d_embed) / np.sqrt(d_embed)
    X_3d = X[:, :, None] * proj[None, :, :]  # (n_samples, n_vars, d_embed)

    var_names = [f"x{i}" for i in range(n_vars)]

    return torch.from_numpy(X_3d).float(), var_names, B


# ── Main demo ────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("CausalNetwork Demo: End-to-End Causal Reasoning Neural Network")
    print("=" * 70)

    # 1. Generate synthetic data
    print("\n[1/4] Generating synthetic data...")
    X, var_names, B_true = generate_linear_gaussian_dag()
    n_vars = len(var_names)
    d_embed = X.shape[-1]
    print(f"  Data shape: {X.shape} (samples×vars×embedding)")
    print(f"  Variables: {var_names}")

    # 2. Create and train causal network
    print("\n[2/4] Creating and training CausalNetwork...")
    net = CausalNetwork(
        d_embed=d_embed,
        n_vars=n_vars,
        var_names=var_names,
        hidden=128,
        rank=32,
        enforce_acyclic=True,
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Training loop
    n_epochs = 300
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Forward pass in training mode
        result = net(X, mode="train")
        loss = result["loss"]

        loss.backward()
        optimizer.step()

        # Update augmented Lagrangian multipliers
        with torch.no_grad():
            A = net.causal_ffnn(X)[0]
            h_val = float(acyclicity_loss(A))

        if (epoch + 1) % 50 == 0:
            net.update_augmented_lagrangian(h_val)
            print(
                f"  Epoch {epoch + 1:3d}/{n_epochs} | Loss: {loss.item():.4f} | "
                f"Align: {result['align_loss'].item():.4f} | h(A): {h_val:.4f}"
            )

    # 3. Build SCM for inference
    print("\n[3/4] Building SCM for inference...")
    scm = net.build_scm(X)
    print("  SCM built successfully")

    # Get learned causal matrix
    A_learned = net.get_causal_matrix()
    print("\n  Top learned causal edges (weight > 0.1):")
    for i in range(n_vars):
        for j in range(n_vars):
            if A_learned[i, j] > 0.1:
                print(f"    {var_names[i]} → {var_names[j]}  w={A_learned[i, j]:.3f}")

    # 4. Intervention prediction
    print("\n[4/4] Performing causal reasoning tasks...")

    # 4a. Intervention effect prediction
    print("\n  a) Intervention effect prediction:")
    print("     do(x0 = 2.0)")

    intervene_result = net(X, mode="intervene", interventions={"x0": 2.0})
    state = intervene_result["state"]

    print("     Resulting state:")
    for i, name in enumerate(var_names):
        if i < 3 or name == "x4":  # Show first 3 and last
            print(f"       {name}: {state[name]:.3f}")

    # 4b. Counterfactual reasoning
    print("\n  b) Counterfactual reasoning:")
    print("     What would x4 be if x0 were 3.0 instead?")

    cf_result = net(X, mode="counterfactual", interventions={"x0": 3.0}, target="x4")
    print(f"     Counterfactual value of x4: {cf_result['value']:.3f}")

    # 4c. Causal planning
    print("\n  c) Causal planning:")
    print("     Find optimal x0 to make x4 = 1.5")

    plan_result = net(X, mode="plan", target={"x4": 1.5}, interv_nodes=["x0"])

    a_opt = plan_result["a_opt"]["x0"]
    print(f"     Optimal intervention: x0 = {a_opt:.3f}")
    print(f"     Final energy: {plan_result['energy']:.6f}")

    # Verify by applying the optimal intervention
    verify_result = net(X, mode="intervene", interventions={"x0": a_opt})
    achieved = verify_result["state"]["x4"]
    print(f"     Achieved x4: {achieved:.3f} (target: 1.5)")

    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
