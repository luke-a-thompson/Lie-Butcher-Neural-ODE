from dysts.flows import Lorenz
import matplotlib.pyplot as plt
import os
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
from training import train_windows

from neural_butcher_ode import NeuralButcherODE, rollout_bnode
from neural_ode import NeuralODE, rollout_node
from typing import Literal


SEED = 42

def main(model: Literal["node", "bnode"]) -> None:
    equation = Lorenz()
    tpts, sol = equation.make_trajectory(500, return_times=True)

    # Build time vector from system timepoints
    ts = jnp.asarray(tpts)
    ys = jnp.asarray(sol)

    if model == "node":
        model = NeuralODE(dim=equation.dimension, vf_width=64, vf_depth=2, stepsize_controller=diffrax.ConstantStepSize(), dt0=ts[1] - ts[0], key=jax.random.PRNGKey(SEED))
        rollout_fn = rollout_node
        model_name = "node"
    elif model == "bnode":
        model = NeuralButcherODE(dim=equation.dimension, vf_width=64, vf_depth=2, key=jax.random.PRNGKey(SEED))
        rollout_fn = rollout_bnode
        model_name = "bnode"
    else:
        raise ValueError(f"Invalid model: {model}")

    # Train on this single trajectory, saving best-by-val checkpoint
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}_best.eqx")

    learned = train_windows(
        ts,
        ys,
        model=model,
        rollout_fn=rollout_fn,
        lr=3e-3,
        window_length=25,
        stride=1,
        batch_size=64,
        train_steps=2000,
        seed=SEED,
        val_fraction=0.2,
        print_every=100,
        checkpoint_path=ckpt_path,
    )

    # Indices for train/validation split (must match val_fraction)
    split_idx = int(ys.shape[0] * 0.8)

    # Roll out on training segment
    ts_train = ts[:split_idx]
    y0_train = ys[0]
    y_model_train = rollout_fn(learned, ts_train, y0_train)

    # Plot train overlay
    os.makedirs("images", exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(ys[:split_idx, 0], ys[:split_idx, 1], c="dodgerblue", label="Real train (x-y)")
    plt.plot(y_model_train[:, 0], y_model_train[:, 1], c="crimson", label="Model train (x-y)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"images/{model_name}_train.png")

    # Load best checkpoint parameters (fallback to current model if missing)
    params, static = eqx.partition(learned, eqx.is_array)
    if os.path.exists(ckpt_path):
        loaded_params = eqx.tree_deserialise_leaves(ckpt_path, params)
        best_model = eqx.combine(loaded_params, static)
    else:
        best_model = learned

    # Roll out best model on validation tail
    ts_tail = ts[split_idx:]
    y0_tail = ys[split_idx]
    y_model_tail = rollout_fn(best_model, ts_tail, y0_tail)

    # Final validation metrics and neat print
    val_true = ys[split_idx:]
    mse = jnp.mean((val_true - y_model_tail) ** 2)
    rmse = jnp.sqrt(mse)
    eps = 1e-8
    mape = jnp.mean(jnp.abs((val_true - y_model_tail) / (jnp.abs(val_true) + eps))) * 100.0
    print("\n===== Final Validation (best checkpoint) =====")
    print(f"Model: {model_name}")
    print(f"Checkpoint: {ckpt_path if os.path.exists(ckpt_path) else 'N/A (used last model)'}")
    print(f"Tail length: {ts_tail.shape[0]} steps")
    print(f"MSE: {float(mse):.6f} | RMSE: {float(rmse):.6f} | MAPE: {float(mape):.4f}%")
    print("============================================\n")

    # Plot tail overlay
    plt.figure(figsize=(6, 4))
    plt.plot(ys[split_idx:, 0], ys[split_idx:, 1], c="dodgerblue", label="Real tail (x-y)")
    plt.plot(y_model_tail[:, 0], y_model_tail[:, 1], c="crimson", label="Model tail (x-y)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"images/{model_name}_tail.png")


if __name__ == "__main__":
    main(model="node")
    main(model="bnode")