from typing import Callable

import jax
import jax.nn.initializers
import jax.numpy as jnp
import jax.random as jr
from flax.struct import dataclass, field

from seher.types import JaxRandomKey, StateEstimator
from seher.apx_arch import MLP

@dataclass
class MLPStateEstimator[Observation, State]:
    mlp: MLP
    obs_to_array: Callable[[Observation], jax.Array] = field(pytree_node=False)
    array_to_state: Callable[[jax.Array], State] = field(pytree_node=False)

    def __call__(
            self,
            carry: None,
            observation: Observation,
            key: JaxRandomKey,
    ) -> State:
        del key, carry
        inpt_arr = self.obs_to_array(observation)
        result_arr = self.mlp(inpt_arr)
        state = self.array_to_state(result_arr)
        return state
    
    __call__.__doc__ = StateEstimator.__call__.__doc__

    def initial_state(self, observation: Observation, key: JaxRandomKey):
        return self(None, observation, key)