from mujoco_playground._src.manipulation.franka_emika_panda.transport_mass import PandaTransportMass, default_config

default = default_config()

print(default)

env = PandaTransportMass(config = default)

