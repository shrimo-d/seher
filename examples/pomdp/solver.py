import dataclasses
from typing import Callable, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import optax
import optuna

from seher.types import POMDP, JaxRandomKey, Policy
from seher.stepper.optax import OptaxOptimizer
from seher.apx_arch import MLP, StaticMLPPolicy
from seher.control.solvers import BaseSolver, _create_optimizer, _create_mlp_policy

from state_estimator import MLPStateEstimator
from joint_belief import *

def _create_mlp_stateestimator(
        obs_to_array: Callable,
        array_to_state: Callable,
        state_dim: int,
        obs_dim: int,
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
            inpt_size=obs_dim, output_size=state_dim, key=key, **defaults
        ),
    )

@dataclasses.dataclass
class JointBeliefPolicySolver(BaseSolver[POMDP]):
    state_dim: int = 1
    obs_to_array: Callable = lambda obs: obs
    state_to_array: Callable = lambda state: state
    array_to_state: Callable = lambda arr: arr
    n_simulations: int = 32
    max_updates: int = 25_000
    max_env_steps: int | None = None

    belief_mlp_kws: dict = dataclasses.field(
        default_factory=lambda: {
            "layer_sizes": [32],
        }
    )
    belief_weight: float = 0.2
    policy_mlp_kws: dict = dataclasses.field(
        default_factory=lambda: {
            "layer_sizes": [32]
        }
    )

    optax_optimizer: str = "adam"
    optax_optimizer_kws: dict = dataclasses.field(
        default_factory=lambda: {"learning_rate": 0.01}
    )
    optimizer_kws: dict = dataclasses.field(default_factory=dict)

    costs: jax.Array | None = dataclasses.field(init=False, default=None)

    def _update_solution_attributes(self, solution):
        self.policy = solution.policy
        self.belief = solution.belief
    
    def _prepare(
            self,
            problem: POMDP,
            key: JaxRandomKey,
    ):
        belief_key, policy_key = jr.split(key)

        sample_state = problem.init(key=key)
        control_dim = problem.empty_control().shape[0]
        obs = problem.emit(sample_state, problem.empty_control(), key)
        obs_dim = self.obs_to_array(obs).shape[0]
        policy_inpt_dim = self.state_to_array(sample_state).shape[0]

        policy = _create_mlp_policy(
            obs_to_array=self.state_to_array,
            obs_dim=policy_inpt_dim,
            control_dim=control_dim,
            key=policy_key,
            **self.policy_mlp_kws,
        )
        belief = _create_mlp_stateestimator(
            obs_to_array=self.obs_to_array,
            array_to_state=self.array_to_state,
            obs_dim=obs_dim,
            state_dim=self.state_dim,
            key=key,
            **self.belief_mlp_kws
        )

        self.policy = policy
        self.belief = belief
        self.prepared = True
        return policy, belief
    
    def solve(
            self,
            problem: POMDP,
            key: JaxRandomKey,
            eval_key: JaxRandomKey | None = None,
    ):
        self.problem = problem

        if eval_key is None:
            eval_key, key = jr.split(key)
            eval_key = cast(JaxRandomKey, eval_key)
        
        if not self.prepared:
            policy, belief = self._prepare(
                problem, key
            )
        else:
            policy = self.policy
            belief = self.belief
        
        if not isinstance(policy, StaticMLPPolicy):
            raise ValueError("policy needs to be of type StaticMLPPolicy")
        if not isinstance(belief, MLPStateEstimator):
            raise ValueError("belief/state estimator needs to be of type MLPStateEstimator")
        
        optimizer = _create_optimizer(
            self.optax_optimizer, self.optax_optimizer_kws, self.optimizer_kws
        )

        joint_belief_optimizer = JointBeliefOptimizer(
            mdp=problem,
            n_simulations=self.n_simulations,
            n_steps=self.steps_per_update,
            optimizer=optimizer,
            belief_weight=self.belief_weight,
            steps_per_init=self.episode_length,
            state_to_array=self.state_to_array
        )

        joint_belief = JointBelief(
            policy=policy,
            belief=belief,
        )
        carry = joint_belief_optimizer.initial_carry(
            sample_parameter=joint_belief
        )
        call_joint_belief = jax.jit(joint_belief_optimizer.__call__)

        solution, aux = self._run_training_loop(
            joint_belief_optimizer,
            carry,
            call_joint_belief,
            key=key,
            eval_key=eval_key,
        )

        self.policy = solution.policy
        self.belief = solution.belief
        self.history = aux.history
        self.is_solved = True

