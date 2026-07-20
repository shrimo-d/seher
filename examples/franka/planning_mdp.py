import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Any, Callable

from flax.struct import dataclass, field

from seher.systems.mujoco_playground import MujocoPlaygroundMDP
from seher.models.state_estimator import StateEstimatorMDPState
import mujoco_playground._src.mjx_env


@dataclass
class RobotPlanningState:
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
class RobotPlanningMDP:
    mdp: MujocoPlaygroundMDP
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
        if isinstance(state, RobotPlanningState):
            return state

        mass = jnp.clip(
            state.est.loc[-1],
            self.mdp.env._config.min_mass,
            self.mdp.env._config.max_mass,
        )

        mjx_model = state.obs.info["mjx_model"].replace(
            body_mass=state.obs.info["mjx_model"].body_mass.at[
                self.mdp.env._payload_id
            ].set(mass)
        )

        obs = state.obs.replace(
            info={
                **state.obs.info,
                "mjx_model": mjx_model,
                "payload_mass": mass,
            }
        )

        return RobotPlanningState(
            base=state.replace(obs=obs),
            planning_mass=mass,
        )
    
    def init(self, key):
        obs0 = self.mdp.init(key)
        se_carry0 = self.estimator.initial_carry()
        prev_control0 = self.empty_control()

        k1, k2 = jr.split(key, 2)
        se_carry1, est0 = self.estimator(se_carry0, obs0, prev_control0, k1)
        latent0 = self.adapter(est0, k2)

        state = StateEstimatorMDPState(
            obs=obs0,
            latent=latent0,
            est=est0,
            se_carry=se_carry1,
        )

        return self.prepare_planning_state(state)
    
    def transit(self, state: RobotPlanningState, control, key):
        k_env, k_est, k_latent = jr.split(key, 3)

        obs1 = self.mdp.transit(state.obs, control, k_env)

        se_carry1, est1 = self.estimator(
            state.se_carry,
            obs1,
            control,
            k_est,
        )
        latent1 = self.adapter(est1, k_latent)

        return RobotPlanningState(
            base=state.base.replace(
                obs=obs1,
                latent=latent1,
                est=est1,
                se_carry=se_carry1,
            ),
            planning_mass=state.planning_mass,
        )
    
    def cost(self, state: RobotPlanningState, control, key):
        base_cost = self.mdp.cost(state.obs, control, key)
        epistemic = jnp.mean(state.est.epistemic_std)

        return base_cost + self.epistemic_penalty * epistemic
    

@dataclass
class VariablePlanningMDP(RobotPlanningMDP):
    penalty_function: Callable = field(pytree_node=False, default=lambda x,y: 0)

    def __hash__(self):
        return id(self)
    
    def __eq__(self, other):
        return self is other

    def init(self, key):
        state = super().init(key)
        return RobotPlanningState(
            base=state.base.replace(last_unc=jnp.array(0.0)),
            planning_mass=state.planning_mass,
        )
    
    def transit(self, state: RobotPlanningState, control, key):
        last_unc = jnp.mean(state.est.epistemic_std)
        new_state = super().transit(state, control, key)
        return RobotPlanningState(
            base=new_state.base.replace(last_unc=last_unc),
            planning_mass=new_state.planning_mass,
        )

    def cost(self, state: RobotPlanningState, control, key):
        base_cost = self.mdp.cost(state.obs, control, key)
        penalty = self.penalty_function(state, control)

        return base_cost + penalty
