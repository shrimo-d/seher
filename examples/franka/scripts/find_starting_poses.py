from mujoco_playground._src.manipulation.franka_emika_panda.transport_mass import PandaTransportMass
import jax.random as jr
import matplotlib.pyplot as plt
import json
from pathlib import Path

#PARAMS
num_states = 300
FRANKA_DIR = Path(__file__).resolve().parents[1]
file_path = FRANKA_DIR / "starting_poses.json"

starting_poses = []
env = PandaTransportMass()

for i in range(num_states):
    state = env.reset(jr.PRNGKey(i))
    img = env.render([state], height=480, width=640)
    plt.imshow(img[0])
    plt.show()
    decision = input("Do you want to keep this as possible starting pose? (y/n)")
    if decision.lower() == "y":
        starting_poses.append(state.data.qpos.tolist())

with open(file_path, "w") as f:
    json.dump(starting_poses, f)
