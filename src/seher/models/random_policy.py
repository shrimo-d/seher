import jax.random as jr
import jax.numpy as jnp
from flax.struct import dataclass, field
from seher.types import MDP


@dataclass
class RandomPolicy:
    mdp: MDP
    z: float = field(pytree_node=False, default=0)
    sigma: float = field(pytree_node=False, default=0.2)

    def __call__(self, carry, obs, control, key):
        z_t1 = jr.normal(key, shape=self.mdp.empty_control().shape)
        z_t1 = self.z + z_t1 * self.sigma
        final = jnp.clip(z_t1, a_min=self.mdp.control_min, a_max=self.mdp.control_max)
        self.replace(z=final)
        return None, final

    def initial_carry(self):
        return None