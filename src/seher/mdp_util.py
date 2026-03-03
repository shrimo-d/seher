import jax
import jax.numpy as jnp
import jax.random as jr

from typing import Callable
from flax.struct import dataclass, field
from seher.types import MDP, State
from seher.models.world_model import WorldModelEnsemble

@dataclass
class WorldModelMDP(MDP):
    original_mdp: MDP
    model: WorldModelEnsemble
    array_to_state: Callable
    uncertainty_weight: float = field(pytree_node=False)

    @property
    def discount(self):
        return self.original_mdp.discount

    def init(self, key):
        return self.original_mdp.init(key)

    def transit(self, state, control, key):
        mean, _ = self.model(state, control, key)
        return self.array_to_state(mean)

    def cost(self, state, control, key):
        base_cost = self.original_mdp.cost(state, control, key)
        _, std = self.model(state, control, key)
        return base_cost + self.uncertainty_weight * std.mean()

    def empty_control(self):
        return self.original_mdp.empty_control()
    
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