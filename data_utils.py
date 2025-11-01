import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


def make_windows(ys: jax.Array, ts: jax.Array, window_length: int, stride: int) -> tuple[jax.Array, jax.Array]:
    num_points = ys.shape[0]
    dim = ys.shape[1]
    starts = jnp.arange(0, max(num_points - window_length + 1, 1), stride)

    def _one_window(start: jax.Array) -> tuple[jax.Array, jax.Array]:
        s = int(start)
        y_win = ys[s : s + window_length]
        t_win = ts[s : s + window_length]
        t_win = t_win - t_win[0]
        return t_win, y_win

    t_list: list[jax.Array] = []
    y_list: list[jax.Array] = []
    for s in starts:
        t_win, y_win = _one_window(s)
        t_list.append(t_win)
        y_list.append(y_win)
    t_windows = jnp.stack(t_list, axis=0) if len(t_list) > 0 else jnp.zeros((1, window_length))
    y_windows = jnp.stack(y_list, axis=0) if len(y_list) > 0 else jnp.zeros((1, window_length, dim))
    return t_windows, y_windows

def dataloader(y_windows: jax.Array, batch_size: int, *, key: jax.Array):
    num_windows = y_windows.shape[0]
    indices = jnp.arange(num_windows)
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        for start in range(0, num_windows, batch_size):
            end = min(start + batch_size, num_windows)
            if start >= end:
                break
            yield y_windows[perm[start:end]]

@eqx.filter_jit
def get_batch(y_windows: jax.Array, batch_size: int, step: jax.Array, *, key: jax.Array) -> jax.Array:
    """Stateless, JIT-friendly batching via per-epoch permutations with fixed batch size.

    Uses wrap-around indexing to always return a batch of shape [batch_size, ...],
    avoiding shape polymorphism and recompilations.
    """
    num_windows = y_windows.shape[0]
    steps_per_epoch = (num_windows + batch_size - 1) // batch_size
    epoch = step // steps_per_epoch
    idx_in_epoch = step % steps_per_epoch

    perm_key = jr.fold_in(key, epoch)
    indices = jnp.arange(num_windows)
    perm = jr.permutation(perm_key, indices)

    start = idx_in_epoch * batch_size
    batch_indices = (start + jnp.arange(batch_size)) % jnp.maximum(num_windows, 1)
    return y_windows[perm[batch_indices]]


