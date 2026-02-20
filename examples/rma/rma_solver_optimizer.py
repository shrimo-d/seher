import dataclasses
import functools
import jax
import jax.numpy as jnp
import jax.random as jr

from typing import cast, Callable
from flax.struct import dataclass, field
from seher.apx_arch import MLP, StaticMLPPolicy
from seher.types import StateEstimator, Policy, State, Observation, Control, Optimizer, POMDP, OptimizerCarry, JaxRandomKey
from seher.control.solvers import BaseSolver, _create_optimizer, _create_mlp_policy

#POMDP: Erste Phase nutzt transit, aber wir brauchen funktion die, die environment data aus dem state nimmt
#Dann Training von Encoder und Base Policy
#Danach Training von adaptation module: aus historz von states und actions latent lernen
@dataclass
class MLPStateEstimator[Observation, State]:
    mlp: MLP
    obs_to_array: Callable[[Observation], jax.Array] = field(pytree_node=False)
    array_to_state: Callable[[jax.Array], State] = field(pytree_node=False)

    def __call__(
            self,
            observation: Observation,
            key: JaxRandomKey,
    ) -> State:
        del key
        inpt_arr = self.obs_to_array(observation)
        result_arr = self.mlp(inpt_arr)
        state = self.array_to_state(result_arr)
        return state
    
    __call__.__doc__ = StateEstimator.__call__.__doc__

    def initial_state(self, observation: Observation, key: JaxRandomKey):
        return self(observation, key)

def _create_mlp_encoder(
        obs_to_array: Callable,
        array_to_state: Callable,
        latent_dim: int,
        priv_info_dim: int,
        key: JaxRandomKey,
        **mlp_kws,
) -> MLPStateEstimator:
    defaults = {
        "layer_sizes": [32],
        "activations": [jnn.soft_sign, lambda x: x]
    }
    defaults.update(mlp_kws)

    return MLPStateEstimator(
        obs_to_array=obs_to_array,
        array_to_state=array_to_state,
        mlp=MLP.make(
            inpt_size=priv_info_dim, output_size=latent_dim, key=key, **defaults
        )
    )

@dataclasses.dataclass
class RMASolver(BaseSolver[POMDP]):
    obs_dim: int = 1
    priv_info_dim: int = 1
    latent_dim: int = 1
    adaptation_history_length: int = 50
    get_priv_info: Callable = lambda state: state
    n_simulations: int = 32
    max_updates: int = 25_000
    max_env_steps: int | None = None

    encoder_mlp_kws: dict = dataclasses.field(
        default_factory=lambda: {
            "layer_sizes": [32],
        }
    )

    policy_mlp_kws: dict = dataclass.field(
        default_factory=lambda: {
            "layer_sizes": [32],
        }
    )

    adaptation_mlp_kws: dict = dataclass.field(
        default_factory=lambda: {
            "layer_sizes": [32],
        }
    )

    optax_optimizer: str = "adam"
    optax_optimizer_kws: dict = dataclasses.field(
        default_factory=lambda: {
            "learning_rate": 0.01,
        }
    )
    optimizer_kws: dict = dataclasses.field(default_factory=None)

    costs: jax.Array | None = dataclasses.field(init=False, default=None)
    phase: bool = False #RMA is two phased, if phase=True it means the second phase of training needs to be used

    def _update_solution_attributes(self, solution):
        if self.phase==False:
            self.encoder = solution.encoder
            self.policy = solution.policy
        else:
            self.adaptation = solution.adaptation
    
    def _prepare(
            self,
            problem: POMDP,
            key: JaxRandomKey,
    ):
        encoder_key, policy_key, adaptation_key, key = jr.split(key, 4)

        sample_state = problem.init(key=key)
        sample_priv = self.get_priv_info(sample_state)
        