import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

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

def true_to_array(state):
    return jnp.asarray(state.info["payload_mass"]).reshape((1,))

def collect_se_dataset(mdp, policy, n_traj: int, n_steps: int, key: jax.Array):
    keys = jr.split(key, n_traj)
    return jax.vmap(lambda k: simulate(mdp=mdp, policy=policy, n_steps=n_steps, key=k))(
        keys
    )

def make_collect_fn(mdp, policy, n_steps):
    @jax.jit
    def collect(keys):
        return jax.vmap(
            lambda k: simulate(mdp=mdp, policy=policy, n_steps=n_steps, key=k)
        )(keys)
    return collect

def extract_arrays(histories, true_to_array: Callable[[Any], jax.Array]):
    obs = histories.states.obs
    act = histories.controls
    true = jax.vmap(lambda traj: jax.vmap(true_to_array)(traj))(histories.states)
    return obs, act, true


def memory_efficient_dataset(mdp, policy, n_traj, n_steps, key, blocksize=500):
    collect = make_collect_fn(mdp, policy, n_steps)

    keys = jr.split(key, n_traj)
    obs_, act_, true_ = [],[],[]

    for start in range(0, n_traj, blocksize):
        block_keys = keys[start:start+blocksize]

        history = collect(block_keys)
        obs, act, true = extract_arrays(history, true_to_array)

        obs_.append(jax.device_get(obs))
        act_.append(jax.device_get(act))
        true_.append(jax.device_get(true))
    
    return (
        jnp.asarray(np.concatenate(obs_, axis=0)),
        jnp.asarray(np.concatenate(act_, axis=0)),
        jnp.asarray(np.concatenate(true_, axis=0)),
    )


def create_mpc_policy(mdp, n_iter, n_plan_steps, n_perturbations, top_k):
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
                n_perturbations=n_perturbations,
                top_k=top_k,
            ),
        ),
    )
    return MPCPolicy(mdp=mdp, planner=stepper)

def create_policy_mix_dataset(mdp, n_traj, n_steps, key, n_plan_steps=5, n_iter=3, n_perturbations=4, top_k=2, std=0.05):
    mpc_policy = create_mpc_policy(mdp, n_iter, n_plan_steps, n_perturbations, top_k)
    neg_policy = RandomNegPolicy(mdp)
    pos_policy = RandomPosPolicy(mdp)
    wak_policy = RandomWalkPolicy(mdp)
    rdm_policy = RandomPolicy(mdp)
    mpc_key, neg_key, pos_key, wak_key, rdm_key, noise_key = jr.split(key, 6)

    mpc_obs, mpc_act, mpc_true = memory_efficient_dataset(mdp, mpc_policy, int(0.2*n_traj), n_steps, mpc_key)

    neg_obs, neg_act, neg_true = memory_efficient_dataset(mdp, neg_policy, int(0.05*n_traj), n_steps, neg_key)

    pos_obs, pos_act, pos_true = memory_efficient_dataset(mdp, pos_policy, int(0.05*n_traj), n_steps, pos_key)

    wak_obs, wak_act, wak_true = memory_efficient_dataset(mdp, wak_policy, int(0.35*n_traj), n_steps, wak_key)

    rdm_obs, rdm_act, rdm_true = memory_efficient_dataset(mdp, rdm_policy, int(0.35*n_traj), n_steps, rdm_key)

    obs = jnp.concatenate([mpc_obs, neg_obs, pos_obs, wak_obs, rdm_obs], axis=0)
    act = jnp.concatenate([mpc_act, neg_act, pos_act, wak_act, rdm_act], axis=0)
    true = jnp.concatenate([mpc_true, neg_true, pos_true, wak_true, rdm_true], axis=0)

    return obs, act, true

