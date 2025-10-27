import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from data_utils import make_windows, dataloader
from collections.abc import Callable


def train_windows(
    ts_full: jax.Array,
    ys_full: jax.Array,
    model: eqx.Module,
    rollout_fn: Callable[[eqx.Module, jax.Array, jax.Array], jax.Array],
    lr: float = 3e-3,
    window_length: int = 50,
    stride: int = 10,
    batch_size: int = 32,
    train_steps: int = 1000,
    seed: int = 5678,
    val_fraction: float = 0.2,
    print_every: int = 100,
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

    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    loader = dataloader(y_windows, batch_size, key=loader_key)

    for step in range(train_steps):
        yi = next(loader)
        t0 = time.time()
        loss, model, opt_state = train_step(model, opt_state, yi)
        t1 = time.time()
        if (step % print_every) == 0 or step == train_steps - 1:
            if y_windows_val.shape[0] > 0:
                y0_val = y_windows_val[:, 0, :]
                ts_val_common = t_windows_val[0]
                y_pred_val = jax.vmap(lambda y0: rollout_fn(model, ts_val_common, y0))(y0_val)
                val_loss = jnp.mean((y_windows_val - y_pred_val) ** 2)
                print(f"Step: {step}, Train loss: {loss}, Val loss: {val_loss}, Time: {t1 - t0}")
            else:
                print(f"Step: {step}, Train loss: {loss}, Time: {t1 - t0}")

    return model


