import diffrax
import equinox as eqx
import jax
from vector_field import MLPVectorField


class NeuralODE(eqx.Module):
    vf: MLPVectorField

    def __init__(self, dim: int = 3, key: jax.Array = None, vf_width: int = 64, vf_depth: int = 2) -> None:
        self.vf = MLPVectorField(in_size=dim, out_size=dim, width_size=vf_width, depth=vf_depth, key=key)

    def __call__(self, ts: jax.Array, y0: jax.Array) -> jax.Array:
        # Autonomous vector field: f(t, y, args) -> vf(y)
        term = diffrax.ODETerm(lambda t, y, args: self.vf(y))
        solution = diffrax.diffeqsolve(
            term,
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys




def rollout_node(model: "NeuralODE", ts: jax.Array, y0: jax.Array) -> jax.Array:
    # Solve on a time grid that starts from 0 for stability
    ts0 = ts - ts[0]
    return model(ts0, y0)


