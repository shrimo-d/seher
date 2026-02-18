import functools
import jax
import jax.numpy as jnp
import jax.random as jr
from typing import cast, Callable
from flax.struct import dataclass, field
from seher.simulate import History, batch_simulate, create_empty_history
from seher.types import Policy, StateEstimator, State, Observation, Control, Optimizer, POMDP, OptimizerCarry, JaxRandomKey

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
class JointBeliefOptimizer[Observation, State, Control](
    Optimizer[
        OptimizerCarry[JointBelief], JointBelief, None, JointBeliefAuxillary
    ]
):
    mdp: POMDP[State, Control, Observation, jax.Array]
    state_to_array: Callable
    n_simulations: int
    n_steps: int
    optimizer: Optimizer[
        OptimizerCarry, JointBelief, None, JointBeliefAuxillary
    ]
    belief_weight: float = field(pytree_node=False)
    steps_per_init: int | None = None

    def initial_carry(self, sample_parameter: JointBelief) -> OptimizerCarry: #noqa: D102
        opt_carry = self.optimizer.initial_carry(sample_parameter) #könnte probleme machen
        keys = jr.split(
            jr.PRNGKey(1), self.n_simulations*self.n_steps
        ).reshape((self.n_simulations, self.n_steps, -1))
        last_history = create_empty_history(
            self.mdp, sample_parameter.policy, keys
        )

        return JointBeliefCarry(  #type: ignore
            current=sample_parameter,
            current_value=jnp.array(float("inf")),
            opt_carry=opt_carry,
            last_history=last_history,
            steps_since_init=jnp.array(0.0),
        )
    
    def objective(
            self,
            joint_belief: JointBelief,
            problem_data: None,
            key: JaxRandomKey,
            carry: JointBeliefCarry,
    ) -> tuple[jax.Array, JointBeliefAuxillary]:
        del problem_data
        policy = joint_belief.policy
        belief = joint_belief.belief
        key, simulate_key = jr.split(key, 2)
        simulate_keys = jr.split(simulate_key, self.n_simulations)

        history = batch_simulate(
            self.mdp,
            policy,
            simulate_keys,
            self.n_steps,
            carry.steps_since_init,
            self.steps_per_init,
            carry.last_history,
            belief,
        )
        
        real_states = self.state_to_array(history.states)
        pred_states = self.state_to_array(history.estimated_states)
        pred_states = pred_states.reshape([self.n_simulations, self.n_steps, -1])

        est_cost = jnp.mean((real_states - pred_states) ** 2)
        cost = history.costs.mean()
        loss = cost + self.belief_weight * est_cost

        return loss, JointBeliefAuxillary(
            history=history,
            loss=loss,
            belief_loss=est_cost,
        )
    
    def __call__(  #noqa: D102
            self,
            carry: OptimizerCarry,
            problem_data: None,
            key: JaxRandomKey,
    ) -> tuple[OptimizerCarry, JointBelief, JointBeliefAuxillary]:
        del problem_data

        carry = cast(JointBeliefCarry, carry)

        objective = functools.partial(self.objective, carry=carry)
        optimizer = self.optimizer.replace(objective=objective)
        opt_carry, joint_belief, aux = optimizer(carry.opt_carry, None, key)

        carry = carry.replace(opt_carry=opt_carry, current=joint_belief)

        return carry, carry.current, aux