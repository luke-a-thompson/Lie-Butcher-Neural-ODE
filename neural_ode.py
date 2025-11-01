import diffrax
import equinox as eqx
import jax
from vector_field import MLPVectorField


class NeuralODE(eqx.Module):
    vf: MLPVectorField
    stepsize_controller: diffrax.AbstractStepSizeController
    dt0: float
    
    def __init__(self, dim: int, vf_width: int, vf_depth: int, stepsize_controller: diffrax.AbstractStepSizeController, dt0: float, key: jax.Array = None) -> None:
        self.dt0 = dt0
        self.vf = MLPVectorField(in_size=dim, out_size=dim, width_size=vf_width, depth=vf_depth, key=key)
        self.stepsize_controller = stepsize_controller

    def __call__(self, ts: jax.Array, y0: jax.Array) -> jax.Array:
        # Autonomous vector field: f(t, y, args) -> vf(y)
        term = diffrax.ODETerm(lambda t, y, args: self.vf(y))
        solution = diffrax.diffeqsolve(
            term,
            diffrax.Heun(),
            t0=ts[0],
            t1=ts[-1],
            dt0=self.dt0,
            y0=y0,
            stepsize_controller=self.stepsize_controller,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys




def rollout_node(model: "NeuralODE", ts: jax.Array, y0: jax.Array) -> jax.Array:
    # Solve on a time grid that starts from 0 for stability
    ts0 = ts - ts[0]
    return model(ts0, y0)


