"""Estimator-aware planning dynamics for the inverted pendulum."""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.random as jr
from flax.struct import dataclass, field

from seher.models.state_estimator import StateEstimatorMDPState
from seher.systems.pendulum_po import PartiallyObservablePendulum


@dataclass
class PendulumPlanningState:
    """Planning state with a mass fixed for one imagined MPC rollout."""

    base: StateEstimatorMDPState
    planning_mass: jax.Array

    @property
    def obs(self):
        return self.base.obs

    @property
    def latent(self):
        return self.base.latent

    @property
    def est(self):
        return self.base.est

    @property
    def se_carry(self):
        return self.base.se_carry

    @property
    def last_unc(self):
        return self.base.last_unc


@dataclass
class PendulumPlanningMDP:
    """Use the current mass estimate in imagined pendulum dynamics."""

    mdp: PartiallyObservablePendulum
    estimator: Any = field(pytree_node=False)
    adapter: Any = field(pytree_node=False)
    epistemic_penalty: float = field(pytree_node=False, default=0.0)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    @property
    def discount(self):
        return self.mdp.discount

    @property
    def control_min(self):
        return self.mdp.control_min

    @property
    def control_max(self):
        return self.mdp.control_max

    def empty_control(self):
        return self.mdp.empty_control()

    def prepare_planning_state(self, state):
        """Inject one clipped estimate and freeze it for this planning call."""

        if isinstance(state, PendulumPlanningState):
            return state

        mass = jnp.clip(
            jnp.asarray(state.est.loc)[-1:],
            self.mdp.min_mass,
            self.mdp.max_mass,
        )
        true = state.obs.true.replace(mass=mass)
        observation = state.obs.replace(true=true)
        return PendulumPlanningState(
            base=state.replace(obs=observation),
            planning_mass=mass,
        )

    def init(self, key):
        k_env, k_estimator, k_latent = jr.split(key, 3)
        observation = self.mdp.init(k_env)
        carry = self.estimator.initial_carry()
        carry, estimate = self.estimator(
            carry,
            observation,
            self.empty_control(),
            k_estimator,
        )
        latent = self.adapter(estimate, k_latent)
        state = StateEstimatorMDPState(
            obs=observation,
            latent=latent,
            est=estimate,
            se_carry=carry,
        )
        return self.prepare_planning_state(state)

    def transit(self, state: PendulumPlanningState, control, key):
        k_environment, k_estimator, k_latent = jr.split(key, 3)
        observation = self.mdp.transit(state.obs, control, k_environment)
        carry, estimate = self.estimator(
            state.se_carry,
            observation,
            control,
            k_estimator,
        )
        latent = self.adapter(estimate, k_latent)
        return PendulumPlanningState(
            base=state.base.replace(
                obs=observation,
                latent=latent,
                est=estimate,
                se_carry=carry,
            ),
            planning_mass=state.planning_mass,
        )

    def cost(self, state: PendulumPlanningState, control, key):
        base_cost = self.mdp.cost(state.obs, control, key)
        epistemic = jnp.mean(state.est.epistemic_std)
        return base_cost + self.epistemic_penalty * epistemic


@dataclass
class VariablePlanningMDP(PendulumPlanningMDP):
    """Planning MDP whose uncertainty penalty is supplied by an experiment."""

    penalty_function: Callable = field(
        pytree_node=False,
        default=lambda state, control: jnp.array(0.0),
    )

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def prepare_planning_state(self, state):
        """Seed relative penalties with the current real uncertainty."""

        if isinstance(state, PendulumPlanningState):
            return state
        state = super().prepare_planning_state(state)
        current_unc = jnp.mean(state.est.epistemic_std)
        return state.replace(base=state.base.replace(last_unc=current_unc))

    def init(self, key):
        return super().init(key)

    def transit(self, state: PendulumPlanningState, control, key):
        last_unc = jnp.mean(state.est.epistemic_std)
        new_state = super().transit(state, control, key)
        return new_state.replace(
            base=new_state.base.replace(last_unc=last_unc),
        )

    def cost(self, state: PendulumPlanningState, control, key):
        base_cost = self.mdp.cost(state.obs, control, key)
        return base_cost + self.penalty_function(state, control)
