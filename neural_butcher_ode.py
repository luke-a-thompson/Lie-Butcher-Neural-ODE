import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from vector_field import MLPVectorField


def push_df(y: jax.Array, v: jax.Array, f: "MLPVectorField") -> jax.Array:
    """Compute Df(y)[v] via forward-mode JVP.

    Df(y)[v] = d/dε f(y + ε v) |_{ε=0}
    """
    _, jvp_out = jax.jvp(f, (y,), (v,))
    return jvp_out


def push_d2f(y: jax.Array, v: jax.Array, w: jax.Array, f: "MLPVectorField") -> jax.Array:
    """Compute D^2 f(y)[v, w] via a JVP over the linear map v -> Df(y)[v].

    D^2 f(y)[v, w] = d/dε ( Df(y + ε w)[v] ) |_{ε=0}
    """
    def df_apply(y_: jax.Array) -> jax.Array:
        return push_df(y_, v, f)

    _, jvp_out = jax.jvp(df_apply, (y,), (w,))
    return jvp_out


def F1_bullet(y: jax.Array, f: "MLPVectorField") -> jax.Array:
    return f(y)


def F2_chain(y: jax.Array, f: "MLPVectorField") -> jax.Array:
    fy = f(y)
    return push_df(y, fy, f)


def F3_chain(y: jax.Array, f: "MLPVectorField") -> jax.Array:
    df_f = F2_chain(y, f)
    return push_df(y, df_f, f)


def F3_bush(y: jax.Array, f: "MLPVectorField") -> jax.Array:
    fy = f(y)
    return push_d2f(y, fy, fy, f)


class BSeriesOrder3Step(eqx.Module):
    f: MLPVectorField
    base_bullet: float
    base_chain: float
    base_c3: float
    base_b3: float

    def __call__(self, y: jax.Array, h: float) -> jax.Array:
        h1 = jnp.asarray(h, dtype=y.dtype)
        h2 = h1 * h1
        h3 = h2 * h1

        c1 = jnp.asarray(self.base_bullet, dtype=y.dtype)
        c2 = jnp.asarray(self.base_chain, dtype=y.dtype)
        c3c = jnp.asarray(self.base_c3, dtype=y.dtype)
        c3b = jnp.asarray(self.base_b3, dtype=y.dtype)

        F1 = F1_bullet(y, self.f)
        F2 = F2_chain(y, self.f)
        F3c = F3_chain(y, self.f)
        F3b = F3_bush(y, self.f)

        return y + h1 * c1 * F1 + h2 * c2 * F2 + h3 * (c3c * F3c + c3b * F3b)


class NeuralButcherODE(eqx.Module):
    """Autonomous Neural ODE integrated via order-3 B-series step (generator frozen)."""

    step: BSeriesOrder3Step

    def __init__(
        self,
        dim: int,
        vf_width: int,
        vf_depth: int,
        dt0: float,
        key: jax.Array = None,
    ) -> None:
        if dim != 3:
            raise ValueError("NeuralButcherODE currently supports 3D states only (got dim != 3)")
        vf_key, _ = jax.random.split(key)
        f = MLPVectorField(in_size=dim, out_size=dim, width_size=vf_width, depth=vf_depth, activation=jnn.softplus, key=vf_key)
        self.dt0 = dt0
        self.step = BSeriesOrder3Step(
            f=f,
            base_bullet=1.0,
            base_chain=0.5,
            base_c3=1.0 / 6.0,
            base_b3=1.0 / 3.0,
        )

    def __call__(self, ts: jax.Array, y0: jax.Array, substeps: int = 1) -> jax.Array:
        assert ts.ndim == 1
        assert y0.shape[0] == 3
        ys = [y0]
        y = y0
        for _ in range(1, ts.shape[0]):
            h = self.dt0 / substeps
            for _ in range(substeps):
                y = self.step(y, h)
            ys.append(y)
        return jnp.stack(ys, axis=0)


def rollout_bnode(model: "NeuralButcherODE", ts: jax.Array, y0: jax.Array, substeps: int = 1) -> jax.Array:
    """Roll out the B-series integrator on a given grid.

    Uses substeps per interval; inverse step achievable by passing negative h via decreasing ts.
    """
    return model(ts, y0, substeps=substeps)



 

