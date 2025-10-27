import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp


class MLPVectorField(eqx.Module):
    mlp: eqx.nn.MLP
    out_scale: jax.Array

    def __init__(
        self,
        *,
        in_size: int,
        out_size: int,
        width_size: int = 64,
        depth: int = 2,
        activation = jnn.softplus,
        key: jax.Array,
    ) -> None:
        self.out_scale = jnp.array(1.0)
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            final_activation=lambda x: x,
            key=key,
        )

    def __call__(self, y: jax.Array) -> jax.Array:
        return self.out_scale * self.mlp(y)




