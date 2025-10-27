import jax
import jax.numpy as jnp
import jax.random as jr


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


