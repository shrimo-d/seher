import argparse
import jax
import jax.numpy as jnp
import jax.random as jr

import sys
from pathlib import Path

from seher.models.state_estimator import FeatureEnsembleLatent
from seher.simulate import simulate

import matplotlib.pyplot as plt

FRANKA_DIR = Path(__file__).resolve().parents[1]
if str(FRANKA_DIR) not in sys.path:
    sys.path.insert(0, str(FRANKA_DIR))

from model_setup import make_components
from planning_mdp import VariablePlanningMDP
from controller_presets import (
    add_controller_arguments,
    resolve_controller_config,
)
from different_penalty_strategies import (
    make_difference_penalty_function, #best weight 32
    make_pos_difference_penalty_function, #best weight 16
    make_static_penalty_function, #best weight 4 or 8 (maybe inbetween)
    make_ratio_penalty_function, #best weight 4 or 8
    make_pos_difference_reward_function,
)
from policies import create_mpc_policy, create_optimizer_from_config

# PARAMS
key=8
n_steps = 400
penalties = [
    (make_static_penalty_function, 0),
    (make_ratio_penalty_function, 6),
    (make_static_penalty_function, 6),
    (make_pos_difference_penalty_function, 16),
]
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

state = wrapped_mdp.init(jr.PRNGKey(key))

locs, stds, costs_real, costs_w_penalty = [], [], [], []

for (penalty, weight) in penalties:

    penalty_mdp = VariablePlanningMDP(
        mdp=mdp,
        adapter=FeatureEnsembleLatent(latent_dim=3),
        estimator=gru,
        penalty_function=penalty(weight)
    )

    policy = create_mpc_policy(
        penalty_mdp,
        args.n_iter,
        args.n_plan_steps,
        optimizer,
    )

    history = simulate(
        mdp=wrapped_mdp,
        policy=policy,
        n_steps=n_steps,
        key=jr.PRNGKey(340),
        initial_state=state,
    )

    locs.append(history.states.latent[:, 0])
    stds.append(history.states.est.epistemic_std[:, 0])
    costs_real.append(history.costs)

    #get augmented costs:
    #state_ = state
    #costs = []
    #for i, control in enumerate(history.controls):
    #    costs.append(penalty_mdp.cost(state_, control, jr.PRNGKey(i)))
    #    state_ = penalty_mdp.transit(state_, control, jr.PRNGKey(i))

    #costs_w_penalty.append(costs)

#PLOTS:
fig, axs = plt.subplots(2, 1, figsize=(14, 14))
axs[0].plot(range(n_steps), jnp.full((n_steps,), fill_value=history.states.obs.info["payload_mass"]), linestyle="--")
for i, (penalty, weight) in enumerate(penalties):
    axs[0].plot(range(n_steps), locs[i], label=f"{penalty.__name__[5:-9] if weight!=0 else "no penalty"}; weight={weight}", color=colors[i])
    axs[0].fill_between(range(n_steps), locs[i]-stds[i], locs[i]+stds[i], color=colors[i], alpha=0.3)
axs[0].legend()
axs[0].set_title("Comparison of estimates with different penalty strategies")

for i, (penalty, weight) in enumerate(penalties):
    axs[1].plot(range(n_steps), costs_real[i], color=colors[i], label=f"{penalty.__name__[5:-9]}; weight={weight}")
axs[1].legend()
axs[1].set_title("Comparison of costs with different penalty strategies")
plt.show()
