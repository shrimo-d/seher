"""Module for simulating systems."""

import abc
import dataclasses
import functools

import jax
import jax.experimental
import jax.lax as jl
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
from flax.struct import dataclass

from seher.types import (
    MDP,
    POMDP,
    Control,
    Cost,
    JaxRandomKey,
    Policy,
    PolicyCarry,
    Carry,
    State,
    Observation,
    StateEstimator,
)
from seher.ui import BaseCLIMetricsCallback


class SimulateCallback[State, PolicyCarry, Control](abc.ABC):
    """Abstract base class for implementing callbacks for simulation.

    During a call to `simulate`, a sub class callback will be called after each
    step with `jax.experimental.io_callback`.

    """

    @abc.abstractmethod
    def __call__(
        self,
        i_step: int,
        state: State,
        estimated_state: State | None,
        policy_carry: PolicyCarry,
        control: Control,
        cost: jax.Array,
    ) -> None:
        """Abstract method that is called during simulate."""
        ...

    def teardown(self) -> None:
        """Do nothing, only fulfill the interface."""
        pass


@dataclasses.dataclass
class RichProgressCallback(
    BaseCLIMetricsCallback,
    SimulateCallback[State, PolicyCarry, Control],
):
    """Class for rich visualization of simulation progress.

    Attributes
    ----------
    total_steps:
        Number of steps this visualisation does.
    description:
        Description to show during the progress printing. Useful if you have
        multiple simulations running consecutively.

    """

    description: str = "Simulating..."
    metrics: tuple[str, ...] = dataclasses.field(
        init=False,
        default_factory=lambda: ("cost",),
    )

    def __call__(
        self,
        i_step: int,
        state: State,
        policy_carry: PolicyCarry,
        control: Control,
        cost: jax.Array,
    ) -> None:
        """Report current simulation state to the command line.

        The state, control and policy carry are ignored. The mean of the cost
        is reported.
        """
        del state, policy_carry, control
        self._update(cost=float(cost.mean()), i_update=i_step)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)


@dataclass
class History[State, Control, Cost, PolicyCarry]:
    """Value class for storing simulation information.

    Attributes
    ----------
    states:
        Of type `State`, but each entry has a prefix shape of `(T+1,)`,
        indexing the time steps.
    controls:
        Of type `Control`, but each entry has a prefix shape of `(T,)`,
        indexing the time steps.
    costs:
        Of type `Cost`, but each entry has a prefix shape of `(T,)`,
        indexing the time steps.
    policy_carries:
        Of type `PolicyCarry`, but each entry has a prefix shape of `(T,)`,
        indexing the time steps.
    estimated_states:
        Of type `State` or None

    """

    states: State
    observations: Observation | None
    controls: Control
    costs: Cost
    policy_carries: PolicyCarry
    state_estimator_carries: Carry | None
    estimated_states: State | None


def simulate(
    mdp: MDP[State, Control, Cost] | POMDP[State, Control, Observation, Cost],
    policy: Policy[State, PolicyCarry, Control],
    n_steps: int,
    key: JaxRandomKey,
    initial_state: State | None = None,
    initial_policy_carry: PolicyCarry | None = None,
    initial_state_estimator_carry: Carry | None = None,
    callback: SimulateCallback[State, PolicyCarry, Control] | None = None,
    state_estimator: StateEstimator[Observation, State, Carry] | None = None,
    jit_policy: bool = True,
    jit_init: bool = True,
    jit_transit: bool = True,
    jit_emit: bool = True,
    jit_estimate: bool = True,
    jit_cost: bool = True,
) -> History[State, Control, Cost, PolicyCarry]:
    """Return history of rolling out `policy` on `mdp`.

    Parameters
    ----------
    mdp:
        System to rollout on (MDP or POMDP).
    policy:
        Actor from which the controls come.
    n_steps:
        Number of time steps to simulate for.
    key:
        RNG for all downstream stochasticity.
    initial_state:
        States to start simulation from. If not given, draw from `mdp.init`.
    initial_policy_carry:
        Policy carries to start policies from. If not given, use the default
        initial policy carry.
    initial_state_estimator_carry:
        State Estimator carries to start state estimators from. None if no
        carry is given.
    callback:
        Will be called after each iteration.
    state_estimator:
        State estimator for POMDPs. If None, no state estimation is performed.
    jit_policy:
        Whether to jit the policy.
    jit_init:
        Whether to jit `mdp.init`.
    jit_transit:
        Whether to jit `mdp.transit`.
    jit_emit:
        Whether to jit `mdp.emit`.
    jit_estimate:
        Whether to jit the state estimator (if applicable).
    jit_cost:
        Whether to jit `mdp.cost`.

    Returns
    -------
    History instance populated with results.

    """
    call_policy = jax.jit(policy.__call__) if jit_policy else policy
    call_init = jax.jit(mdp.init) if jit_init else mdp.init
    call_transit = jax.jit(mdp.transit) if jit_transit else mdp.transit
    call_emit = jax.jit(mdp.emit) if hasattr(mdp, "emit") and jit_emit else \
                mdp.emit if hasattr(mdp, "emit") else None
    call_estimate = jax.jit(state_estimator.__call__) if state_estimator is \
                    not None and jit_estimate else state_estimator
    call_cost = jax.jit(mdp.cost) if jit_cost else mdp.cost

    init_key, key = jr.split(key)

    if initial_state is None:
        initial_state = call_init(key=init_key)

    if initial_policy_carry is None:
        initial_policy_carry = policy.initial_carry()

    initial_control = mdp.empty_control()
    initial_observation = mdp.initial_observation(initial_state) if hasattr(mdp, "initial_observation") \
                          else None

    if state_estimator is None:
        initial_estimate = initial_state
    else:
        initial_estimate = state_estimator.initial_state(initial_observation, init_key)

    def scan_step(carry, _):
        i_step, true_state, observation, state_estimate, policy_carry, se_carry, control, key = carry

        policy_key, transit_key, emit_key, cost_key, key = jr.split(key, 5)

        if state_estimate is None:
            state_estimate = true_state

        policy_carry, control = call_policy(
            carry=policy_carry,
            obs=state_estimate,
            control=control,
            key=policy_key,
        )

        cost = call_cost(state=true_state, control=control, key=cost_key)
        state_p1 = call_transit(state=true_state, control=control, key=transit_key)

        if call_emit is not None:
            observation_p1 = call_emit(state=true_state, control=control, key=emit_key)
            if state_estimator is not None:
                se_carry, state_estimate = call_estimate(se_carry, observation_p1, key)
            else:
                state_estimate = state_p1
        else:
            observation_p1 = None
            state_estimate = state_p1

        if callback is not None:
            jax.experimental.io_callback(
                callback, None, i_step, true_state, observation_p1, state_estimate, policy_carry,
                se_carry, control, cost
            )

        new_carry = (i_step + 1, state_p1, observation_p1, state_estimate, policy_carry, se_carry, control, key)
        outputs = (state_p1, observation_p1, state_estimate, control, cost, policy_carry, se_carry)

        return new_carry, outputs

    initial_carry = (
        0,
        initial_state,
        initial_observation,
        initial_estimate,
        initial_policy_carry,
        initial_state_estimator_carry,
        initial_control,
        key,
    )

    _, (states, observations, estimated_states, controls, costs, policy_carries, se_carries) = jax.lax.scan(
        scan_step,
        initial_carry,
        None,  # Not needed since we're just iterating n_steps times.
        length=n_steps,
    )

    result = History(
        states=states,
        observations=observations,
        controls=controls,
        costs=costs,
        policy_carries=policy_carries,
        state_estimator_carries=se_carries,
        estimated_states=estimated_states
    )

    if callback is not None:
        jax.experimental.io_callback(callback.teardown, None)

    return result


def init_or_persist(
    mdp: MDP[State, Control, Cost] | POMDP[State, Control, Observation, Cost],
    policy: Policy[State, PolicyCarry, Control],
    last_history: History[State, Control, Cost, PolicyCarry] | None,
    steps_since_init: jax.Array,
    steps_per_init: int,
    key: JaxRandomKey,
    state_estimator: StateEstimator[State, Observation, Carry] | None = None,
) -> tuple[State, PolicyCarry]:
    """Initialize new episode or persist from last history."""
    key, init_key = jr.split(key)

    def init(key):
        se_carry = None
        if state_estimator is not None:
            se_carry = state_estimator.initial_carry()
        return mdp.init(key), policy.initial_carry(), se_carry

    def persist(key):
        if last_history is None:
            raise ValueError(".last_history must be set")
        states = jt.map(lambda x: x[-1], last_history.states)
        policy_carries = jt.map(lambda x: x[-1], last_history.policy_carries)
        se_carries = jt.map(lambda x: x[-1], last_history.state_estimator_carries)

        return states, policy_carries, se_carries

    initial_state, initial_policy_carry, initial_se_carry = jl.cond(
        (steps_since_init % steps_per_init != 0),
        persist,
        init,
        init_key,
    )

    return initial_state, initial_policy_carry, initial_se_carry


@functools.partial(jax.vmap, in_axes=(None, None, 0, None))
@functools.partial(jax.vmap, in_axes=(None, None, 0, None))
def create_empty_history(
    mdp: MDP[State, Control, Cost] | POMDP[State, Control, Observation, Cost],
    policy: Policy[State, PolicyCarry, Control],
    key: JaxRandomKey,
    state_estimator: StateEstimator[State, Observation, Carry] | None = None
) -> History[State, Control, Cost, PolicyCarry]:
    """Create empty history structure for initialization."""
    # Create a zero-like cost structure by running cost once and mapping to
    # zeros.
    temp_state = mdp.init(key)
    temp_control = mdp.empty_control()
    temp_cost = mdp.cost(temp_state, temp_control, key)
    zero_cost = jax.tree_util.tree_map(jnp.zeros_like, temp_cost)

    if state_estimator is None:
        temp_obs=None
        estimated_states = None
        initial_se_carry = None
    else:
        temp_obs = mdp.initial_observation(temp_state)
        initial_se_carry = state_estimator.initial_carry()
        estimated_states = state_estimator(initial_se_carry, temp_obs, key)

    return History(
        states=mdp.init(key),
        observations=temp_obs,
        controls=mdp.empty_control(),
        policy_carries=policy.initial_carry(),
        state_estimator_carries=initial_se_carry,
        costs=zero_cost,
        estimated_states=estimated_states,
    )


@functools.partial(jax.vmap, in_axes=(None, None, 0, None, None, None, 0, None))
def batch_simulate(
    mdp: MDP[State, Control, Cost] | POMDP[State, Control, Observation, Cost],
    policy: Policy[State, PolicyCarry, Control],
    key: JaxRandomKey,
    n_steps: int,
    steps_since_init: jax.Array,
    steps_per_init: int | None,
    last_history: History[State, Control, Cost, PolicyCarry] | None,
    state_estimator: StateEstimator[Observation, State, Carry] | None,
) -> History[State, Control, Cost, PolicyCarry]:
    """Simulate policy on MDP with batching and episode persistence."""
    if steps_per_init is not None:
        initial_state, initial_policy_carry, initial_se_carry = init_or_persist(
            mdp=mdp,
            policy=policy,
            state_estimator=state_estimator,
            last_history=last_history,
            steps_per_init=steps_per_init,
            steps_since_init=steps_since_init,
            key=key,
        )

    else:
        initial_state = None
        initial_policy_carry = None
        initial_se_carry = None

    history = simulate(
        policy=policy,
        mdp=mdp,
        key=key,
        n_steps=n_steps,
        initial_state=initial_state,
        initial_policy_carry=initial_policy_carry,
        initial_state_estimator_carry=initial_se_carry,
        state_estimator=state_estimator,
    )
    return history
