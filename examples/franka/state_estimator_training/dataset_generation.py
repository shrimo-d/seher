import jax
import jax.numpy as jnp
import jax.random as jr

from flax.struct import dataclass
from typing import Callable, Any
import functools
import optax

from seher.types import MDP
from seher.simulate import simulate
from seher.models.random_policy import RandomPolicy, RandomWalkPolicy
from seher.control.mpc import MPCPolicy
from seher.control.stepper_planner import StepperPlanner
from seher.ars import ars_value_and_grad
from seher.stepper.optax import OptaxOptimizer

@dataclass
class RandomPosPolicy:
    mdp: MDP

    def __call__(self, carry, obs, control, key):
        final = jr.uniform(
            key,
            shape=self.mdp.empty_control().shape,
            minval=0.0,
            maxval=self.mdp.control_max,
        )
        return None, final
    
    def initial_carry(self):
        return None


@dataclass
class RandomNegPolicy:
    mdp: MDP

    def __call__(self, carry, obs, control, key):
        final = jr.uniform(
            key,
            shape=self.mdp.empty_control().shape,
            minval=self.mdp.control_min,
            maxval=0.0,
        )
        return None, final
    
    def initial_carry(self):
        None

def collect_se_dataset(mdp, policy, n_traj: int, n_steps: int, key: jax.Array):
    keys = jr.split(key, n_traj)
    return jax.vmap(lambda k: simulate(mdp=mdp, policy=policy, n_steps=n_steps, key=k))(
        keys
    )

def extract_arrays(histories, true_to_array: Callable[[Any], jax.Array]):
    obs = histories.states
    act = histories.controls
    true = jax.vmap(lambda traj: jax.vmap(true_to_array)(traj))(histories.states)
    return obs, act, true

def create_mpc_policy(mdp, n_iter, n_plan_steps):
    stepper = StepperPlanner(
        mdp=mdp,
        n_iter=n_iter,
        n_plan_steps=n_plan_steps,
        warm_start=True,
        optimizer=OptaxOptimizer(
            objective=None,
            optimizer=optax.adam(0.03),
            value_and_grad=functools.partial(
                ars_value_and_grad,
                std=0.2,
                n_perturbations=32,
                top_k=8,
            ),
        ),
    )
    return MPCPolicy(mdp=mdp, planner=stepper)

def create_policy_mix_dataset(mdp, n_traj, n_steps, key):
    mpc_policy = create_mpc_policy(mdp, 13, 20)
    neg_policy = RandomNegPolicy(mdp)
    pos_policy = RandomPosPolicy(mdp)
    wak_policy = RandomWalkPolicy(mdp)
    rdm_policy = RandomPolicy(mdp)
    mpc_key, neg_key, pos_key, wak_key, rdm_key = jr.split(key, 5)

    mpc_history = collect_se_dataset(mdp, mpc_policy, int(0.2*n_traj), n_steps, mpc_key)
    neg_history = collect_se_dataset(mdp, neg_policy, int(0.2*n_traj), n_steps, neg_key)
    pos_history = collect_se_dataset(mdp, pos_policy, int(0.2*n_traj), n_steps, pos_key)
    wak_history = collect_se_dataset(mdp, wak_policy, int(0.2*n_traj), n_steps, wak_key)
    rdm_history = collect_se_dataset(mdp, rdm_policy, int(0.2*n_traj), n_steps, rdm_key)

    histories = [mpc_history, neg_history, pos_history, wak_history, rdm_history]

    return jax.tree.map(
        lambda *xs: jnp.concatenate(xs, axis=0),
        *histories
    )

