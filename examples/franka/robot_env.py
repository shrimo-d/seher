from mujoco_playground._src.manipulation.franka_emika_panda.transport_mass import PandaTransportMass
from mujoco_playground._src import mjx_env
from mujoco_playground._src.mjx_env import State
import mujoco_playground as mp
from seher.systems.mujoco_playground import MujocoPlaygroundMDP
from flax.struct import dataclass
from seher.types import JaxRandomKey

import jax
import jax.numpy as jnp

@dataclass
class RobotEnv(MujocoPlaygroundMDP):
    """This is essentially a 'reimplementation' of the reset method of the
    PandaTransportMass environment baked into the seher wrapper.
    The PandaTransportMass environment can start in self colliding states.
    This environment with its init method uses a curated list of 'legal' states.
    """

    starting_poses: tuple = (1,0)

    def init(self, key: JaxRandomKey) -> mp.State:  # noqa: D102
        key, rng_mass, rng_spawn = jax.random.split(key, 3)
        starting_poses = self.get_starting_poses()
        pose = jax.random.choice(rng_spawn, starting_poses)

        data = mjx_env.make_data(
            self.env._mj_model,
            qpos=pose,
            qvel=jnp.zeros(self.env._mjx_model.nv, dtype=float),
            ctrl=pose,
            impl=self.env._mjx_model.impl.value,
            nconmax=self.env._config.nconmax,
            njmax=self.env._config.njmax,
        )

        metrics = {
            **{k: 0.0 for k in self.env._config.reward_config.scales.keys()},
        }
        info = {
            "rng": key,
            "reached_box": 0.0,
            "start_pos": pose,
            "last_action": pose,
        }
        obs = self.env._get_obs(data, info)
        reward, done = jnp.zeros(2)
        pre_state = State(data, obs, reward, done, metrics, info)
        mass = jax.random.uniform(rng_mass, minval=self.env._config.min_mass, maxval=self.env._config.max_mass)
        state = self.env.set_payload_mass(pre_state, mass)
        return state
    
    def get_starting_poses(self):
        return jnp.asarray(self.starting_poses)