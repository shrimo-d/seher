import jax
import jax.numpy as jnp
import jax.random as jr

from mujoco_playground._src.manipulation.franka_emika_panda.transport_mass import PandaTransportMass, default_config
from seher.systems.mujoco_playground import MujocoPlaygroundMDP

default = default_config()

print(default)

env = PandaTransportMass(config = default)

mdp = MujocoPlaygroundMDP(env)

state = mdp.init(jr.PRNGKey(0))

print(state)

print(mdp.transit(state, mdp.empty_control(), jr.PRNGKey(2)))

print(state.obs)
print(state.info["payload_mass"])
print(mdp.control_min, mdp.control_max)