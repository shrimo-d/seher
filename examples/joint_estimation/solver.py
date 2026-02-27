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
from seher.control.solvers import PolicyGradientSolver, _create_optimizer, _create_mlp_policy
from seher.control.policy_search.policy_gradient import PolicyGradientOptimizer
from seher.control.tanh_gaussian_policy_mdp import (
    SimulationPolicy,
    TanhGaussianPolicyControl,
    TanhGaussianPolicyState,
)

from se import StochasticMLPStateEstimatorSlidingWindow
from separated_joint_belief import *

def _create_mlp_stateestimator(
        obs_to_array,
        array_to_state,
        window_size,
        obs_dim,
        latent_dim,
        state_dim,
        key,
        **mlp_kws,
):
    latent_defaults = {
        "layer_sizes": [32],
        "activations": [jnn.soft_sign, lambda x: x]
    }
    mu_defaults = {
        "layer_sizes": [32],
        "activations": [jnn.soft_sign, lambda x: x]
    }
    sigma_defaults = {
        "layer_sizes": [32],
        "activations": [jnn.soft_sign, lambda x: x]
    }

    latent_defaults.update(mlp_kws["latent_mlp_kws"])
    mu_defaults.update(mlp_kws["mu_mlp_kws"])
    sigma_defaults.update(mlp_kws["sigma_mlp_kws"])

    latent_key, mu_key, sigma_key = jr.split(key, 3)
    
    return StochasticMLPStateEstimatorSlidingWindow(
        obs_to_array=obs_to_array,
        obs_dim=obs_dim,
        array_to_state=array_to_state,
        window_size=window_size,
        state_dim=state_dim,
        latent_mlp=MLP.make(
            inpt_size=(window_size+1)*obs_dim,
            output_size=latent_dim,
            key=latent_key,
            **latent_defaults
        ),
        mu_mlp=MLP.make(
            inpt_size=latent_dim,
            output_size=state_dim,
            key=mu_key,
            **mu_defaults
        ),
        sigma_mlp=MLP.make(
            inpt_size=latent_dim,
            output_size=state_dim,
            key=sigma_key,
            **sigma_defaults
        ),
    )

def universal_state_to_array(state_to_array):
    def wrapper(state):
        state = getattr(state, "original_state", state)
        return state_to_array(state)
    return wrapper

def universal_obs_to_array(obs_to_array):
    def wrapper(obs):
        obs = getattr(obs, "original_observation", obs)
        return obs_to_array(obs)
    return wrapper

@dataclass
class RandomPolicy[Observation, Carry, Control]:
    """Random Policy mit Batch-Unterstützung für JAX."""

    control_dim: int
    control_low: float = -1.0
    control_high: float = 1.0

    def initial_carry(self):
        return None

    def __call__(
        self,
        carry: None,
        obs: Observation,
        control: Control,
        key: JaxRandomKey,
    ):
        """Gibt zufällige Controls zurück (unterstützt Batches)."""
        del carry, obs, control
        return None, jr.uniform(
            key,
            shape=(self.control_dim,),
            minval=self.control_low,
            maxval=self.control_high
        )


@dataclasses.dataclass
class StateEstimatorPolicySolver(PolicyGradientSolver):
    """Policy Gradient Solver that uses a StateEstimatorOptimizer.
    
    SE does a number of updates before a policy update happens with
    the newly trained se.
    
    """
    state_dim: int = 1
    obs_to_array: Callable = lambda obs: obs
    state_to_array: Callable = lambda state: state
    array_to_state: Callable = lambda arr: arr
    n_simulations: int = 32
    max_updates: int = 25_000
    se_updates_per_policy: int = 5
    max_env_steps: int | None = None

    state_estimator_latent_dim: int = 2
    state_estimator_window_size: int = 5
    state_estimator_mlp_kws: dict = dataclasses.field(
        default_factory=lambda: {
            "latent_mlp_kws": {
                "layer_sizes": [32],
            },
            "mu_mlp_kws": {
                "layer_sizes": [32],
            },
            "sigma_mlp_kws": {
                "layer_sizes": [32],
            }
        }
    )
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
        self._gaussian_policy = solution.policy
        self.policy = SimulationPolicy(
            solution.policy,
            self.problem.control_min, self.problem.control_max,
        )
        self.state_estimator = solution.belief
    
    def _prepare(
            self,
            problem: POMDP,
            key: JaxRandomKey,
    ):
        state_estimator_key, policy_key = jr.split(key)

        sample_state = problem.init(key=key)
        control_dim = problem.empty_control().shape[0]
        obs = problem.initial_observation(sample_state)
        obs_dim = self.obs_to_array(obs).shape[0]
        policy_inpt_dim = self.state_to_array(sample_state).shape[0]

        belief = _create_mlp_stateestimator(
            obs_to_array=universal_obs_to_array(self.obs_to_array),
            array_to_state=self.array_to_state,
            window_size=self.state_estimator_window_size,
            obs_dim=obs_dim,
            latent_dim=self.state_estimator_latent_dim,
            state_dim=self.state_dim,
            key=state_estimator_key,
            **self.state_estimator_mlp_kws
        )

        policy = _create_mlp_policy(
            obs_to_array=universal_state_to_array(self.state_to_array),
            obs_dim=policy_inpt_dim,
            control_dim=2*control_dim,
            key=policy_key,
            **self.policy_mlp_kws,
        )

        def array_to_gaussian_control(arr):
            return TanhGaussianPolicyControl(
                loc=arr[:control_dim],
                inv_softplus_scale=arr[control_dim:],
            )
        gaussian_policy = StaticMLPPolicy(
            mlp=policy.mlp,
            obs_to_array=universal_state_to_array(self.state_to_array),
            array_to_control=array_to_gaussian_control,
        )
        self._gaussian_policy = gaussian_policy

        self.random_policy = RandomPolicy(
            control_dim=control_dim,
            control_low=problem.control_min,
            control_high=problem.control_max,
        )


        self.policy = SimulationPolicy(
            gaussian_policy, problem.control_min, problem.control_max
        )
        self.belief = belief
        self.prepared = True
        return gaussian_policy, belief
    
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
        if not isinstance(belief, StochasticMLPStateEstimatorSlidingWindow):
            raise ValueError("belief/state estimator needs to be of type StochasticMLPStateEstimatorSlidingWindow")
        
        se_optimizer = _create_optimizer(
            self.optax_optimizer, self.optax_optimizer_kws, self.optimizer_kws
        )
        policy_optimizer = _create_optimizer(
            self.optax_optimizer, self.optax_optimizer_kws, self.optimizer_kws
        )

        joint_belief_optimizer = SeparatedJointBeliefOptimizer(
            mdp=problem,
            n_simulations=self.n_simulations,
            n_steps=self.steps_per_update,
            state_to_array=self.state_to_array,
            policy_optimizer=PolicyGradientOptimizer(
                mdp=problem,
                gaussian_policy=policy,
                n_simulations=self.n_simulations,
                n_steps=self.steps_per_update,
                optimizer=policy_optimizer,
                steps_per_init=self.episode_length,
                state_estimator=belief,
            ),
            se_optimizer=SEOptimizer(
                mdp=problem,
                policy=self.random_policy,
                state_to_array=self.state_to_array,
                n_simulations=self.n_simulations,
                n_steps=self.steps_per_update,
                optimizer=se_optimizer,
                steps_per_init=self.episode_length,
            ),
            se_updates_per_policy=self.se_updates_per_policy,
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

        self._gaussian_policy = solution.policy
        self.policy = SimulationPolicy(
            solution.policy, problem.control_min, problem.control_max
        )
        self.belief = solution.belief
        self.history = aux.history
        self.se_costs = aux.belief_loss
        self.is_solved = True