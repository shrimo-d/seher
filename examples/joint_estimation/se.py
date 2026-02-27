from typing import Callable

import jax
import jax.nn.initializers
import jax.numpy as jnp
import jax.random as jr
from flax.struct import dataclass, field

from seher.types import JaxRandomKey, StateEstimator
from seher.apx_arch import MLP

@dataclass
class SECarry:
    last_obs: jax.Array
    mus: jax.Array
    sigmas: jax.Array


@dataclass
class StochasticMLPStateEstimatorSlidingWindow[Observation, State]:
    latent_mlp: MLP
    mu_mlp: MLP
    sigma_mlp: MLP
    state_dim: int = field(pytree_node=False)
    obs_dim: int = field(pytree_node=False)
    obs_to_array: Callable[[Observation], jax.Array] = field(pytree_node=False)
    array_to_state: Callable[[jax.Array], State] = field(pytree_node=False)
    window_size: int = field(pytree_node=False, default=5)

    def __call__(
            self,
            carry: SECarry,
            observation: Observation,
            key: JaxRandomKey,
    ) -> State:
        inpt_arr = self.obs_to_array(observation)

        if carry is None:
            carry = self.initial_carry()

        new_last_obs = jnp.concatenate(
            [carry.last_obs[1:], inpt_arr[None, :]], axis=0
        )

        window = carry.last_obs
   
        window = window.reshape(-1)
        inpt_arr = jnp.concatenate([inpt_arr, window], axis=-1)
        latent_arr = self.latent_mlp(inpt_arr)

        mu_arr = self.mu_mlp(latent_arr)
        log_sigma_arr = self.sigma_mlp(latent_arr)
        sigma_arr = jnp.exp(log_sigma_arr)

        epsilon = jr.normal(key, mu_arr.shape)
        result_arr = mu_arr + sigma_arr * epsilon

        state = self.array_to_state(result_arr)

        new_carry = carry.replace(
            last_obs=new_last_obs,
            mus=mu_arr,
            sigmas=sigma_arr,
        )


        return new_carry, state
    
    __call__.__doc__ = StateEstimator.__call__.__doc__

    def initial_state(self, observation: Observation, key: JaxRandomKey):
        return self.array_to_state(jnp.zeros(self.state_dim,))
    
    def initial_carry(self):
        return SECarry(
            last_obs=jnp.zeros([self.window_size, self.obs_dim]),
            mus=jnp.zeros([self.state_dim]),
            sigmas=jnp.ones([self.state_dim]),
        )