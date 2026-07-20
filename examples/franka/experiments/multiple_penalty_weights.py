import argparse
import jax
import jax.numpy as jnp
import jax.random as jr

import sys
from pathlib import Path

from seher.models.state_estimator import FeatureEnsembleLatent
from seher.simulate import simulate

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

FRANKA_DIR = Path(__file__).resolve().parents[1]
if str(FRANKA_DIR) not in sys.path:
    sys.path.insert(0, str(FRANKA_DIR))

from controller_presets import (
    add_controller_arguments,
    resolve_controller_config,
)
from model_setup import make_components
from planning_mdp import VariablePlanningMDP
from different_penalty_strategies import (
    make_difference_penalty_function, #best weight 32
    make_pos_difference_penalty_function, #best weight 16
    make_static_penalty_function, #best weight 4 or 8 (maybe inbetween)
    make_ratio_penalty_function, #best weight 4 or 8
    make_mixed_penalty_function,
)
from policies import create_mpc_policy, create_optimizer_from_config

# PARAMS
n_steps = 400
penalty_func = make_static_penalty_function #make_static_penalty_function
epistemic_penalties = [0.0, 1.0, 2.0, 4.0, 6.0] #[0.0, 1.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=n_steps)
    add_controller_arguments(parser, default_preset="ars_reference")
    args = parser.parse_args()
    resolve_controller_config(
        args,
        default_preset="ars_reference",
        parser=parser,
    )
    return args


args = parse_args()
n_steps = args.n_steps
mdp, gru, wrapped_mdp, _ = make_components()
optimizer = create_optimizer_from_config(args.controller_config)

locs = []
stds = []
costs = []

for penalty in epistemic_penalties:

    epipen_mdp = VariablePlanningMDP(
        mdp=mdp,
        adapter=FeatureEnsembleLatent(latent_dim=3),
        estimator=gru,
        penalty_function=penalty_func(penalty)
    )

    epipen_policy = create_mpc_policy(
        epipen_mdp,
        args.n_iter,
        args.n_plan_steps,
        optimizer,
    )

    state = wrapped_mdp.init(jr.PRNGKey(420))

    history_epipen = simulate(
        mdp=wrapped_mdp,
        policy=epipen_policy,
        n_steps=n_steps,
        key=jr.PRNGKey(400),
        initial_state=state,
    )

    po_states_epipen = history_epipen.states.obs
    po_latent_epipen = history_epipen.states.latent

    #Create epipen gif
    state_list_epipen = [jax.tree.map(lambda x: x[i], po_states_epipen)
                        for i in range(po_states_epipen.reward.shape[0])]
    imgs_epipen = mdp.env.render(state_list_epipen, height=480, width=640)

    fig, ax = plt.subplots()
    im = ax.imshow(imgs_epipen[0])
    ax.axis("off")

    def update(frame):
        im.set_array(frame)
        return [im]

    ani = FuncAnimation(
        fig,
        update,
        frames=imgs_epipen,
        blit=True
    )
    ani.save(
        f"rollout_epipen_{penalty}.gif",
        writer=PillowWriter(fps=200)
    )

    locs.append(po_latent_epipen[:, 0])
    stds.append(history_epipen.states.est.epistemic_std[:, 0])
    costs.append(history_epipen.costs)





fig, ax = plt.subplots(figsize=(14,7))
ax.plot(range(n_steps), jnp.full((n_steps,), fill_value=po_states_epipen.info["payload_mass"]), linestyle="--")
for i, (loc, std) in enumerate(zip(locs, stds)):
    ax.plot(range(n_steps), loc, label=f"penalty weight {epistemic_penalties[i]}", color=colors[i])
    ax.fill_between(range(n_steps), loc-std, loc+std, alpha=0.3, color=colors[i])
ax.legend()
ax.set_title("Effect on State Estimates of varying Penalty Weights")
plt.show()


fig, ax = plt.subplots(figsize=(14,7))
for i, cost in enumerate(costs):
    ax.plot(range(n_steps), cost, color=colors[i], label=f"penalty weight {epistemic_penalties[i]}")
ax.legend()
ax.set_title("Effect on Costs of varying Penalty Weights")
plt.show()
