import jax
import jax.numpy as jnp
import jax.random as jr

import dataclasses
from typing import Callable, cast, Any
from seher.types import (
    StateEstimator,
    Policy,
    JaxRandomKey,
    MDP,
    StateCritic,
    LatentAdapter
)
from seher.control.solvers import ActorCriticSolver, _create_optimizer
from seher.control.policy_search.actor_critic import ActorCriticOptimizer, ActorCritic
from seher.apx_arch import StaticMLPCritic
from seher.simulate import History, simulate
from seher.models.state_estimator import StateEstimatorMDP
from Code.seher.examples.joint_estimation.se_optimizer import StateEstimatorOptimizer, StateEstimatorTrainingBatch, StateEstimatorParameters


@dataclasses.dataclass
class EstimatorTrajectoryBuffer:
    """Fixed-size trajectory buffer in pure jax arrays.
    
    Shapes
    ------
    observations:   [capacity, T, obs_dim]
    controls:       [capacity, T, control_dim]
    true_states:    [capacity, T, state_dim]
    
    Notes
    -----
    - This is deliberately array-only, not list-based, so the add/sample path can
    remain jax-friendly
    - All trajectories stored in this buffer must have the same rollout length.
    """
    observations: jax.Array
    controls: jax.Array
    true_states: jax.Array
    size: jax.Array
    ptr: jax.Array
    capacity: int

    @classmethod
    def empty(
        cls,
        capacity: int,
        rollout_steps: int,
        obs_dim: int,
        control_dim: int,
        state_dim: int,
    ) -> "EstimatorTrajectoryBuffer":
        return cls(
            observations=jnp.zeros((capacity, rollout_steps, obs_dim)),
            controls=jnp.zeros((capacity, rollout_steps, control_dim)),
            true_states=jnp.zeros((capacity, rollout_steps, state_dim)),
            size=jnp.array(0, dtype=jnp.int32),
            ptr=jnp.array(0, dtype=jnp.int32),
            capacity=capacity,
        )

def add_to_trajectory_buffer(
        buffer: EstimatorTrajectoryBuffer,
        observations: jax.Array,
        controls: jax.Array,
        true_states: jax.Array,
) -> EstimatorTrajectoryBuffer:
    """Add a batch of trajectories to the ring buffer.
    
    Parameters
    ----------
    observations
        Shape [B, T, obs_dim]
    controls
        Shape [B, T, control_dim]
    true_states
        Shape [B, T, state_dim]
    
    """
    B = observations.shape[0]
    capacity = buffer.capacity

    idxs = (buffer.ptr + jnp.arange(B)) % capacity

    new_observations = buffer.observations.at[idxs].set(observations)
    new_controls = buffer.controls.at[idxs].set(controls)
    new_true_states = buffer.true_states.at[idxs].set(true_states)

    new_ptr = (buffer.ptr + B) % capacity
    new_size = jnp.minimum(buffer.size + B, capacity)

    return dataclasses.replace(
        buffer,
        observations=new_observations,
        controls=new_controls,
        true_states=new_true_states,
        ptr=new_ptr,
        size=new_size,
    )

def reconstruct_carry(
        estimator: StateEstimator,
        obs_seq: jax.Array,
        control_seq: jax.Array,
        t: jax.Array,
        key: JaxRandomKey,
):
    """Reconstruct estimator carry by replacing a trajectory prefix."""
    T = obs_seq.shape[0]
    keys = jr.split(key, T)

    def step(carry, xs):
        i, obs, control, k = xs
        def do_update(c):
            c_new, _ = estimator(c, obs, control, k)
            return c_new
        carry = jax.lax.cond(
            i<t,
            do_update,
            lambda c:c,
            carry,
        )
        return carry, None
    
    init_carry = estimator.initial_carry()
    idxs = jnp.arange(T)
    final_carry, _ = jax.lax.scan(
        step,
        init_carry,
        (idxs, obs_seq, control_seq, keys),
    )
    return final_carry

def build_estimator_training_batch(
        buffer: EstimatorTrajectoryBuffer,
        estimator: StateEstimator,
        batch_size: int,
        key: JaxRandomKey,
) -> StateEstimatorTrainingBatch:
    """Sample one-step estimator training batch from trajectory buffer."""
    if int(buffer.size) == 0:
        raise ValueError("Cannot sample from empty Buffer!")
    
    key_traj, key_time, key_carry = jr.split(key, 3)
    traj_idxs = jr.randint(
        key_traj,
        shape=(batch_size,),
        minval=0,
        maxval=buffer.size,
    )
    
    T = buffer.controls.shape[1]
    ts = jr.randint(
        key_time,
        shape=(batch_size,),
        minval=0,
        maxval=T,
    )

    obs_seq = buffer.observations[traj_idxs]
    control_seq = buffer.controls[traj_idxs]
    true_state_seq = buffer.true_states[traj_idxs]

    observations = obs_seq[jnp.arange(batch_size), ts]
    controls = control_seq[jnp.arange(batch_size), ts]
    targets = true_state_seq[jnp.arange(batch_size), ts]

    carry_keys = jr.split(key_carry, batch_size)

    def per_sample_carry(obs_one, ctrl_one, t, k):
        return reconstruct_carry(
            estimator=estimator,
            obs_seq=obs_one,
            control_seq=ctrl_one,
            t=t,
            key=k,
        )
    
    init_carry = jax.vmap(per_sample_carry)(
        obs_seq,
        control_seq,
        ts,
        carry_keys,
    )

    return StateEstimatorTrainingBatch(
        observations=observations,
        controls=controls,
        targets=targets,
        init_carry=init_carry,
    )

@dataclasses.dataclass
class JointActorCriticStateEstimator:
    actor: Policy
    critic: StateCritic
    target_critic: StateCritic | None
    estimator: StateEstimator

@dataclasses.dataclass
class JointTrainingAuxiliary:
    actor_critic_aux: Any
    state_estimator_aux: Any
    history: History | None

@dataclasses.dataclass
class JointActorCriticStateEstimatorSolver(ActorCriticSolver):
    """Actor-Critic solver with outer-loop estimator updates.
    
    One outer iteration does:
        1) collect trajectories with current wrapped MDP
        2) push them into estimator trajectory buffer
        3) run several estimator updates
        4) rebuild wrapped MDP with new estimator
        5) run one actor-critic update block
    
    """
    base_problem: MDP | None = dataclasses.field(init=False, default=None)
    estimator: StateEstimator | None = None
    state_estimator_optimizer: StateEstimatorOptimizer | None = None

    estimator_buffer_capacity: int = 256
    estimator_batch_size: int = 64
    rollout_steps_per_update: int = 32

    estimator_updates_per_ac_update: int = 8

    estimator_obs_to_array: Callable = lambda x: x
    estimator_state_to_array: Callable = lambda x: x
    estimator_control_to_array: Callable = lambda x: x

    uncertainty_cost_fn: Callable | None = None
    adapter: LatentAdapter | None = None

    trained_estimator: StateEstimator | None = dataclasses.field(
        init=False, default=None
    )
    estimator_buffer: EstimatorTrajectoryBuffer | None = dataclasses.field(
        init=False, default=None
    )

    def _update_solution_attributes(self, solution):
        """Update actor, critic, policy, estimator."""
        self.actor = solution.actor
        self.critic = solution.critic
        self.policy = solution.actor
        self.trained_estimator = solution.estimator
    
    def _wrap_problem(
        self,
        problem: MDP,
        estimator: StateEstimator,
    ) -> MDP:
        """Wrap MDP with current estimator."""
        return StateEstimatorMDP(
            original_mdp=problem,
            estimator=estimator,
            penalty_fn=self.uncertainty_cost_fn,
            adapter=self.adapter
        )
    
    def _prepare(
        self,
        problem: MDP,
        policy_init_key: JaxRandomKey,
        critic_init_key: JaxRandomKey,
    ):
        """Reuse ActorCriticSolver network init."""
        actor, critic = super()._prepare(
            problem, policy_init_key, critic_init_key
        )
        return actor, critic
    
    def _create_actor_critic_optimizer(
        self,
        problem: MDP,
    ) -> ActorCriticOptimizer:
        optimizer = _create_optimizer(
            self.optax_optimizer,
            self.optax_optimizer_kws,
            self.optimizer_kws,
        )
        return ActorCriticOptimizer(
            mdp=problem,
            n_simulations=self.n_simulations,
            n_steps=self.steps_per_update,
            optimizer=optimizer,
            critic_weight=self.critic_weight,
            td_lambda=self.td_lambda,
            steps_per_init=self.episode_length,
        )
    
    def _extract_estimator_arrays(
            self,
            history: History,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Convert simulated rollout history to raw estimator arrays.
        
        self.estimator_obs_to_array and self.estimator_state_to_array
        should obviously convert the StateEstimatorMDP State into arrays."""
        observations = jax.vmap(
            jax.vmap(self.estimator_obs_to_array)
        )(history.states)

        true_states = jax.vmap(
            jax.vmap(self.estimator_state_to_array)
        )(history.states)
        controls = jax.vmap(
            jax.vmap(self.estimator_control_to_array)
        )(history.controls)

        return observations, controls, true_states
    
    def _collect_trajectory_batch(
            self,
            problem: MDP,
            actor: Policy,
            key: JaxRandomKey,
    ) -> tuple[History, jax.Array, jax.Array, jax.Array]:
        """Collect fresh trajectories and extract estimator arrays."""
        keys = jr.split(key, self.n_simulations)

        # Erstmal bewusst OHNE jax.jit, damit es robust läuft.
        history = jax.vmap(
            lambda k: simulate(
                mdp=problem,
                policy=actor,
                n_steps=self.rollout_steps_per_update,
                key=k,
            )
        )(keys)

        observations, controls, true_states = self._extract_estimator_arrays(history)
        return history, observations, controls, true_states
    
    def _initialize_estimator_buffer(
            self,
            problem: MDP,
    ) -> EstimatorTrajectoryBuffer:
        """Create empty fixed-size estimator trajectory buffer from problem dims."""
        sample_state = problem.init(key=jr.PRNGKey(0))
        obs_dim = self.estimator_obs_to_array(sample_state).shape[0]
        state_dim = self.estimator_state_to_array(sample_state).shape[0]
        control_dim = self.estimator_control_to_array(problem.empty_control()).shape[0]

        return EstimatorTrajectoryBuffer.empty(
            capacity=self.estimator_buffer_capacity,
            rollout_steps=self.rollout_steps_per_update,
            obs_dim=obs_dim,
            control_dim=control_dim,
            state_dim=state_dim,
        )
    
    def _run_training_loop(
            self,
            actor_critic_optimizer: ActorCriticOptimizer,
            state_estimator_optimizer: StateEstimatorOptimizer,
            initial_ac_carry,
            initial_se_carry,
            key: JaxRandomKey,
            eval_key: JaxRandomKey,
            wrapped_problem: MDP,
    ):
        """Joint training loop."""
        del eval_key

        ac_carry = initial_ac_carry
        se_carry = initial_se_carry

        # Erstmal NICHT jitten.
        call_actor_critic = actor_critic_optimizer.__call__
        call_state_estimator = state_estimator_optimizer.__call__

        total_env_steps = 0
        buffer = self._initialize_estimator_buffer(wrapped_problem)

        for i in range(self.max_updates):
            key, collect_key, est_key, ac_key = jr.split(key, 4)

            history, observations, controls, true_states = self._collect_trajectory_batch(
                wrapped_problem,
                ac_carry.current.actor,
                collect_key,
            )

            buffer = add_to_trajectory_buffer(
                buffer,
                observations=observations,
                controls=controls,
                true_states=true_states,
            )

            se_aux = None
            se_solution = se_carry.current
            for k in range(self.estimator_updates_per_ac_update):
                batch_key = jr.fold_in(est_key, k)
                train_batch = build_estimator_training_batch(
                    buffer=buffer,
                    estimator=se_carry.current.estimator,
                    batch_size=self.estimator_batch_size,
                    key=batch_key,
                )
                step_key = jr.fold_in(est_key, 1000 + k)
                se_carry, se_solution, se_aux = call_state_estimator(
                    se_carry,
                    train_batch,
                    step_key,
                )

            wrapped_problem = self._wrap_problem(
                cast(MDP, self.base_problem),
                se_solution.estimator,
            )
            self.problem = wrapped_problem
            actor_critic_optimizer = self._create_actor_critic_optimizer(wrapped_problem)
            call_actor_critic = actor_critic_optimizer.__call__

            ac_carry, ac_solution, ac_aux = call_actor_critic(
                ac_carry,
                None,
                ac_key,
            )

            solution = JointActorCriticStateEstimator(
                actor=ac_solution.actor,
                critic=ac_solution.critic,
                target_critic=ac_solution.target_critic,
                estimator=se_solution.estimator,
            )
            aux = JointTrainingAuxiliary(
                actor_critic_aux=ac_aux,
                state_estimator_aux=se_aux,
                history=history,
            )

            self._update_solution_attributes(solution)
            total_env_steps += self.n_simulations * self.steps_per_update

            eval_history = None
            if (
                self.updates_per_eval is not None
                and i % self.updates_per_eval == 0
            ):
                eval_history = self.simulate(
                    n_steps=self.episode_length,
                    n_simulations=self.eval_n_simulations,
                )

            for callback in self.callbacks:
                callback(
                    i,
                    train_history=history,
                    eval_history=eval_history,
                    aux=aux,
                )

            if (
                self.max_env_steps is not None
                and total_env_steps >= self.max_env_steps
            ):
                break

        for callback in self.callbacks:
            if hasattr(callback, "teardown"):
                callback.teardown()

        self.estimator_buffer = buffer
        return solution, aux
    
    def solve(
            self,
            problem: MDP,
            key: JaxRandomKey,
            eval_key: JaxRandomKey | None = None,
            policy_init_key: JaxRandomKey | None = None,
            critic_init_key: JaxRandomKey | None = None,
    ):
        """Solve joint actor-critic + state-estimator training."""
        if self.estimator is None:
            raise ValueError("estimator must be set.")
        if self.state_estimator_optimizer is None:
            raise ValueError("state_estimator_optimizer must be set.")

        if eval_key is None:
            eval_key, key = jr.split(key)
            eval_key = cast(JaxRandomKey, eval_key)
        if policy_init_key is None:
            policy_init_key, key = jr.split(key)
            policy_init_key = cast(JaxRandomKey, policy_init_key)
        if critic_init_key is None:
            critic_init_key, key = jr.split(key)
            critic_init_key = cast(JaxRandomKey, critic_init_key)
        
        self.base_problem = problem
        wrapped_problem = self._wrap_problem(problem, self.estimator)
        self.problem = wrapped_problem

        if not self.prepared:
            actor, critic = self._prepare(
                wrapped_problem, policy_init_key, critic_init_key,
            )
        else:
            actor = self.actor
            critic = self.critic
        
        if not isinstance(critic, StaticMLPCritic):
            raise ValueError("critic needs to be of type StaticMLPCritic")
        
        target_critic = (
            jax.tree_util.tree_map(lambda x: x, critic)
            if self.polyak_step_size is not None
            else None
        )

        actor_critic_optimizer = self._create_actor_critic_optimizer(
            wrapped_problem
        )
        ac_solution = ActorCritic(
            actor=actor,
            critic=critic,
            target_critic=target_critic,
        )
        ac_carry = actor_critic_optimizer.initial_carry(
            sample_parameter=ac_solution
        )
        se_params = StateEstimatorParameters(estimator=self.estimator)
        se_carry = self.state_estimator_optimizer.initial_carry(
            sample_parameter=se_params,
        )

        solution, aux = self._run_training_loop(
            actor_critic_optimizer=actor_critic_optimizer,
            state_estimator_optimizer=self.state_estimator_optimizer,
            initial_ac_carry=ac_carry,
            initial_se_carry=se_carry,
            key=key,
            eval_key=eval_key,
            wrapped_problem=wrapped_problem,
        )

        self.actor = solution.actor
        self.critic = solution.critic
        self.policy = solution.actor
        self.trained_estimator = solution.estimator
        self.history = aux.history
        self.is_solved = True