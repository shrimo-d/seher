import jax.random as jr
from flax.struct import dataclass
from seher.types import MDP


@dataclass
class RandomPolicy:
    mdp: MDP

    def __call__(self, carry, obs, control, key):
        return None, jr.uniform(
            key,
            shape=self.mdp.empty_control().shape,
            minval=self.mdp.control_min,
            maxval=self.mdp.control_max,
        )

    def initial_carry(self):
        return None