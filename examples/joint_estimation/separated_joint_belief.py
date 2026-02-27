import jax
import jax.numpy as jnp
import jax.random as jr
from typing import cast, Callable
import functools
from jax.tree_util import tree_map
from flax.struct import dataclass, field
from seher.simulate import History, batch_simulate, create_empty_history
from seher.types import Policy, StateEstimator, State, Observation, Control, Optimizer, POMDP, OptimizerCarry, JaxRandomKey, PolicyCarry
from seher.control.tanh_gaussian_policy_mdp import SimulationPolicy

@dataclass
class JointBeliefAuxillary:
    history: History
    loss: jax.Array
    belief_loss: jax.Array

@dataclass
class JointBelief:
    policy: Policy
    belief: StateEstimator

@dataclass
class JointBeliefCarry:
    current: JointBelief
    current_value: jax.Array | None
    opt_carry: OptimizerCarry
    last_history: History | None
    steps_since_init: jax.Array

@dataclass
class SEOptimizerCarry:
    current: StateEstimator
    current_value: jax.Array
    opt_carry: OptimizerCarry
    last_history: History
    steps_since_init: jax.Array

@dataclass
class SeparatedJointBeliefCarry:
    se_carry: SEOptimizerCarry
    policy_carry: PolicyCarry
    current: JointBelief

@dataclass
class SEAuxillary:
    history: History
    loss: jax.Array

@dataclass
class SEOptimizer(Optimizer):
    mdp: POMDP
    state_to_array: callable
    n_simulations: int
    n_steps: int
    optimizer: Optimizer
    policy: Policy
    steps_per_init: int | None = None

    def initial_carry(self, se: StateEstimator) -> OptimizerCarry:
        keys = jr.split(
            jr.PRNGKey(1), self.n_simulations*self.n_steps
            ).reshape((self.n_simulations, self.n_steps, -1))
        last_history = create_empty_history(self.mdp, self.policy, keys, se)
        opt_carry = self.optimizer.initial_carry(se)
        return SEOptimizerCarry(
            current=se,
            current_value=jnp.array(float("inf")),
            last_history=last_history,
            steps_since_init=jnp.array(0.0),
            opt_carry=opt_carry,
        )
    
    def objective(self, se: StateEstimator, problem_data, key, carry):
        key, simulate_key = jr.split(key, 2)
        simulate_keys = jr.split(simulate_key, self.n_simulations)

        history = batch_simulate(
            self.mdp,
            self.policy,
            simulate_keys,
            self.n_steps,
            carry.steps_since_init,
            self.steps_per_init,
            carry.last_history,
            se
        )
        real_states = self.state_to_array(history.states)
                #this is only work for pendulum. needs to be reworked in future:
        dim = real_states.shape[-1] - 1
        mus = history.state_estimator_carries.mus.reshape((self.n_simulations, self.n_steps, dim))
        mus = jnp.concatenate([jnp.cos(mus[:,:,0:1]), jnp.sin(mus[:,:,0:1]), mus[:,:,1:2]], axis=-1)
        sigmas = history.state_estimator_carries.sigmas.reshape((self.n_simulations, self.n_steps, dim))
        sigmas = jnp.concatenate([sigmas[:,:,0:1], sigmas[:,:,0:1], sigmas[:,:,1:2]], axis=-1)

        diff = real_states - mus
        nll = 0.5*((diff/sigmas) ** 2 + 2 * jnp.log(sigmas) + jnp.log(2*jnp.pi))
        loss = jnp.mean(nll)
    
        return loss, SEAuxillary(history=history, loss=loss)
    
    def __call__(self, carry, problem_data, key):
        objective = functools.partial(self.objective, carry=carry)
        optimizer = self.optimizer.replace(objective=objective)
        opt_carry, se, aux = optimizer(carry.opt_carry, None, key)
        steps = carry.steps_since_init + 1
        return carry.replace(opt_carry=opt_carry, current=se, last_history=aux.history, steps_since_init=steps), se, aux
    
@dataclass
class SeparatedJointBeliefOptimizer(Optimizer):
    mdp: POMDP
    state_to_array: Callable
    n_simulations: int
    n_steps: int
    se_optimizer: SEOptimizer
    policy_optimizer: Optimizer
    se_updates_per_policy: int = 5

    def initial_carry(self, sample_parameter: JointBelief) -> OptimizerCarry:
        se_carry = self.se_optimizer.initial_carry(sample_parameter.belief)
        policy_carry = self.policy_optimizer.initial_carry(sample_parameter.policy)
        return SeparatedJointBeliefCarry(
            se_carry=se_carry,
            policy_carry=policy_carry,
            current=sample_parameter,
        )
    
    def __call__(self, carry, problem_data, key):
        se_carry, policy_carry = carry.se_carry, carry.policy_carry
        se_loss = 0.0
        policy = SimulationPolicy(
            carry.policy_carry.current,
            self.mdp.control_min, self.mdp.control_max
        )

        
        frozen_policy = tree_map(jax.lax.stop_gradient, policy)
        se_optimizer = self.se_optimizer.replace(policy=frozen_policy)

        for _ in range(self.se_updates_per_policy):
            key, subkey = jr.split(key)
            se_carry, se, se_aux = se_optimizer(se_carry, None, subkey)
            se_loss += se_aux.loss
        
        key, subkey = jr.split(key)
        frozen_se = tree_map(jax.lax.stop_gradient, se)

        policy_optimizer = self.policy_optimizer.replace(state_estimator=frozen_se)
        policy_carry, policy, aux = policy_optimizer(policy_carry, None, subkey)

        new_carry = carry.replace(
            se_carry=se_carry,
            policy_carry=policy_carry,
        )

        return new_carry, JointBelief(
            policy=policy, belief=se
        ), JointBeliefAuxillary(
            history=aux.history,
            loss=aux.loss,
            belief_loss=se_loss/self.se_updates_per_policy
        )