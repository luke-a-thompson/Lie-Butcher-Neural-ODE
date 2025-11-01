import time
import os
from collections import deque
from tqdm import trange

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from data_utils import make_windows, get_batch
from collections.abc import Callable


def initialize_jax_compilation_cache(cache_dir: str | None = None) -> None:
    """Initialize JAX persistent compilation cache if available.

    Uses environment variable JAX_COMPILATION_CACHE_DIR when set, otherwise
    defaults to a local 'jax_cache' directory in the current working directory.
    """
    try:
        from jax.experimental.compilation_cache import compilation_cache as cc  # type: ignore
    except Exception:
        return

    if cache_dir is None:
        cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR", os.path.join(os.getcwd(), "jax_cache"))

    try:
        # Support both older and newer APIs
        if hasattr(cc, "initialize_cache"):
            cc.initialize_cache(cache_dir)  # type: ignore[attr-defined]
        elif hasattr(cc, "set_cache_dir"):
            cc.set_cache_dir(cache_dir)  # type: ignore[attr-defined]
    except Exception:
        # Best-effort initialization; ignore failures.
        pass


# Initialize cache on import
initialize_jax_compilation_cache()

def train_windows(
    ts_full: jax.Array,
    ys_full: jax.Array,
    model: eqx.Module,
    rollout_fn: Callable[[eqx.Module, jax.Array, jax.Array], jax.Array],
    lr: float,
    window_length: int,
    stride: int,
    batch_size: int,
    train_steps: int,
    seed: int,
    val_fraction: float = 0.2,
    print_every: int = 100,
    checkpoint_path: str | None = None,
) -> eqx.Module:
    key = jr.PRNGKey(seed)
    loader_key = key

    split_idx = max(int((1.0 - val_fraction) * ys_full.shape[0]), window_length)
    ts_train = ts_full[:split_idx]
    ys_train = ys_full[:split_idx]
    ts_val = ts_full[split_idx:]
    ys_val = ys_full[split_idx:]

    t_windows, y_windows = make_windows(ys_train, ts_train, window_length, stride)
    if ys_val.shape[0] >= window_length:
        t_windows_val, y_windows_val = make_windows(ys_val, ts_val, window_length, stride)
    else:
        t_windows_val = jnp.zeros_like(t_windows)
        y_windows_val = jnp.zeros_like(y_windows)

    ts_common = t_windows[0]

    optim = optax.adam(learning_rate=lr, b1=0.9, b2=0.999)

    @eqx.filter_value_and_grad
    def loss_fn(model: eqx.Module, yi: jax.Array) -> jax.Array:
        y0_batch = yi[:, 0, :]

        def do_rollout(y0: jax.Array) -> jax.Array:
            return rollout_fn(model, ts_common, y0)

        y_pred = jax.vmap(do_rollout)(y0_batch)
        return jnp.mean((yi - y_pred) ** 2)

    @eqx.filter_jit
    def train_step(model: eqx.Module, opt_state: object, yi: jax.Array) -> tuple[jax.Array, eqx.Module, object]:
        loss, grads = loss_fn(model, yi)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    @eqx.filter_jit
    def eval_step(model: eqx.Module, yi_val: jax.Array, ts_val_common: jax.Array) -> jax.Array:
        y0_val = yi_val[:, 0, :]
        y_pred_val = jax.vmap(lambda y0: rollout_fn(model, ts_val_common, y0))(y0_val)
        return jnp.mean((yi_val - y_pred_val) ** 2)

    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    # Prepare a fixed base key for stateless batching
    batch_key = loader_key

    # JIT compilation warmup: run a single dummy train step to compile
    if y_windows.shape[0] > 0:
        yi_warmup = get_batch(y_windows, batch_size, jnp.array(0), key=batch_key)
        warmup_loss, _, _ = train_step(model, opt_state, yi_warmup)
        jax.block_until_ready(warmup_loss)

    # JIT compilation warmup for eval step (if validation windows exist)
    if y_windows_val.shape[0] > 0:
        ts_val_common_warm = t_windows_val[0]
        yi_val_warmup = y_windows_val[: min(batch_size, y_windows_val.shape[0])]
        warmup_val_loss = eval_step(model, yi_val_warmup, ts_val_common_warm)
        jax.block_until_ready(warmup_val_loss)

    # Best-checkpoint tracking
    best_val_loss: float = float("inf")
    if checkpoint_path is not None:
        ckpt_dir = os.path.dirname(checkpoint_path)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)

    bar = trange(train_steps, desc="Training", leave=True)
    step_times = deque(maxlen=print_every)
    last_val_loss_float: float | None = None
    for step in bar:
        yi = get_batch(y_windows, batch_size, jnp.array(step), key=batch_key)
        t0 = time.time()
        loss, model, opt_state = train_step(model, opt_state, yi)
        t1 = time.time()

        # Compute validation loss only at print_every cadence (to save time)
        val_loss = None
        if y_windows_val.shape[0] > 0 and ((step % print_every) == 0 or step == train_steps - 1):
            ts_val_common = t_windows_val[0]
            val_loss = eval_step(model, y_windows_val, ts_val_common)

            # Save best checkpoint by validation loss
            val_loss_float: float = float(val_loss)
            if val_loss_float < best_val_loss and checkpoint_path is not None:
                best_val_loss = val_loss_float
                params, static = eqx.partition(model, eqx.is_array)
                eqx.tree_serialise_leaves(checkpoint_path, params)
            last_val_loss_float = val_loss_float

        # Update progress bar with 3dp metrics
        train_loss_float: float = float(loss)
        dt_float: float = float(t1 - t0)
        step_times.append(dt_float)
        # Mean time per epoch (estimated over last print_every steps)
        if len(step_times) > 0:
            mean_time_per_epoch: float = (sum(step_times) / len(step_times)) * print_every
        else:
            mean_time_per_epoch = 0.0

        postfix: dict[str, str] = {
            "train": f"{train_loss_float:.3f}",
            "t/ep": f"{mean_time_per_epoch:.3f}s",
        }
        if last_val_loss_float is not None:
            postfix["val"] = f"{last_val_loss_float:.3f}"
            postfix["best_val"] = f"{best_val_loss:.3f}"
        bar.set_postfix(postfix)

    return model


