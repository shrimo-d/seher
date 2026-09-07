import jax
import jax.numpy as jnp
import jax.random as jr

from typing import Callable
from flax.struct import dataclass
from seher.types import MDP, State

    
@dataclass
class NoiseWrapperState:
    original_state: State
    noisy_state: State


@dataclass
class NoiseWrapperMDP(MDP):
    mdp: MDP
    add_noise_to_state: Callable

    @property
    def discount(self):
        return self.mdp.discount
    
    def init(self, key):
        ini_state = self.mdp.init(key)
        new_state = self.add_noise(ini_state, key)
        return NoiseWrapperState(original_state=ini_state,
                                 noisy_state=new_state)
    
    def transit(self, state, control, key):
        new_state = self.mdp.transit(state.original_state, control, key)
        return NoiseWrapperState(original_state=new_state,
                                 noisy_state=self.add_noise(new_state, key))
    
    def cost(self, state, control, key):
        return self.mdp.cost(state.original_state, control, key)

    def empty_control(self):
        return self.mdp.empty_control()
    
    def add_noise(self, orig_state, key):
        return self.add_noise_to_state(orig_state, key)