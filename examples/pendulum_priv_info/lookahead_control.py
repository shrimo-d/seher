import jax
import jax.numpy as jnp
import jax.random as jr

import functools
import optax

from flax.struct import dataclass
from typing import Callable, Any

from seher.models.state_estimator import (
    StateEstimatorMDP,
    FeatureEnsembleLatent,
)
from seher.systems.pendulum_po import PartiallyObservablePendulum, POPendulumTrueState, POPendulumObservation, POPendulumState
from seher.systems.pendulum import render
from seher.control.stepper_planner import StepperPlanner
from seher.stepper.optax import OptaxOptimizer
from seher.ars import ars_value_and_grad
from seher.control.mpc import MPCPolicy
from seher.simulate import simulate
from planning_mdp import EstimatedPendulumPlanningMDP
from load_models import maybe_load_estimator

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec

#VARIABLES

mass = 2.5 # For single plot
masses = [0.5, 0.6875, 0.875, 1.0625, 1.25, 1.4375, 1.625, 2.0]
n_steps = 200
epistemic_penalty = 15.0
start_upright = True
n_iter=13
n_plan_steps=20

#Functions and Classes
@dataclass
class ConstantExcitationMPCPolicy:
    mpc: MPCPolicy
    noise_std: float = 0.5

    def initial_carry(self):
        return self.mpc.initial_carry()
    
    def __call__(self, carry, obs, control, key):
        k_pol, k_noise = jr.split(key)

        carry, action = self.mpc(carry, obs, control, k_pol)
        noise = jr.normal(k_noise, shape=action.shape) * self.noise_std
        noisy_action = action + noise
        return carry, noisy_action


def create_stepper(mdp, n_iter, n_plan_steps):
    return StepperPlanner(
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


def make_state_upright(state):
    true = state.obs.true.replace(
        angle=jnp.array((0.,)),
        velocity=jnp.array((0.,))
    )
    obs = state.obs.replace(true=true)
    return state.replace(obs=obs)


def change_state_mass(state, mass):
    new_true = state.obs.true.replace(mass=jnp.array((mass,)))
    new_obs = state.obs. replace(true=new_true)
    return state.replace(obs=new_obs)


def plot_se_traj(ax, mass_estimates, color, epistemic_std, label):
    ax.plot(range(len(mass_estimates)), mass_estimates, color=color, label=label)
    ax.fill_between(
        range(len(mass_estimates)),
        mass_estimates - epistemic_std,
        mass_estimates + epistemic_std,
        alpha=0.25,
        color=color
    )


#Run experiment
mdp = PartiallyObservablePendulum(min_mass=0.5, max_mass=2.5)

gru = maybe_load_estimator(Path("./examples/pendulum_priv_info"), "det_gru_ensemble")

wrapped_mdp = StateEstimatorMDP(
    original_mdp=mdp,
    estimator=gru,
    adapter=FeatureEnsembleLatent(latent_dim=12),
    concatenate_obs_est=False,
    obs_to_array=lambda state: jnp.concatenate([state.obs.angle, state.obs.velocity])
)

plan_mdp_epipen = EstimatedPendulumPlanningMDP(mdp=mdp, epistemic_penalty=epistemic_penalty, estimator=gru, adapter=FeatureEnsembleLatent(latent_dim=12))
plan_mdp_nopen = EstimatedPendulumPlanningMDP(mdp=mdp, epistemic_penalty=0.0, estimator=gru, adapter=FeatureEnsembleLatent(latent_dim=12))

planner_epipen = create_stepper(plan_mdp_epipen, n_iter, n_plan_steps)
planner_nopen = create_stepper(plan_mdp_nopen, n_iter, n_plan_steps)



policy_epipen = MPCPolicy(mdp=plan_mdp_epipen, planner=planner_epipen)
policy_nopen = MPCPolicy(mdp=plan_mdp_nopen, planner=planner_nopen)
policy_const_excitation = ConstantExcitationMPCPolicy(mpc=policy_nopen, noise_std=0.7)


ini_state = wrapped_mdp.init(jr.PRNGKey(67))
ini_state = change_state_mass(ini_state, mass)
upright_state = make_state_upright(ini_state)

state = upright_state if start_upright else ini_state

history_epipen = simulate(
    mdp=wrapped_mdp,
    policy=policy_epipen,
    n_steps=n_steps,
    key=jr.PRNGKey(200),
    initial_state=state,
)
history_nopen = simulate(
    mdp=wrapped_mdp,
    policy=policy_nopen,
    n_steps=n_steps,
    key=jr.PRNGKey(200),
    initial_state=state,
)
history_excit = simulate(
    mdp=wrapped_mdp,
    policy=policy_const_excitation,
    n_steps=n_steps,
    key=jr.PRNGKey(200),
    initial_state=state,
)

po_states_epipen = history_epipen.states.obs
po_latent_epipen = history_epipen.states.latent

po_states_nopen = history_nopen.states.obs
po_latent_nopen = history_nopen.states.latent

po_states_excit = history_excit.states.obs
po_latent_excit = history_excit.states.latent

angles_epipen = po_states_epipen.true.angle_normed
angles_nopen = po_states_nopen.true.angle_normed
angles_excit = po_states_excit.true.angle_normed


#PLOTS
fig, ax = plt.subplots(3,2,figsize=(10,10))
#epistemic penalty plot
render(angles_epipen, ax[0, 0])
ax[0,1].plot(jnp.arange(n_steps), jnp.full((n_steps,), fill_value=po_states_epipen.true.mass[0]), linestyle="--")
plot_se_traj(ax[0,1], po_latent_epipen[:, 3], "orange", history_epipen.states.est.epistemic_std[:,3], "epistemic_pen")
ax[0,1].spines["top"].set_visible(False)
ax[0,1].spines["right"].set_visible(False)
ax[0,1].grid(visible=True, alpha=0.2)
ax[0,1].set_title("State Estimates with epistemic penalty stepper planner")
#no penalty plot
render(angles_nopen, ax[1,0])
ax[1,1].plot(jnp.arange(n_steps), jnp.full((n_steps,), fill_value=po_states_nopen.true.mass[0]), linestyle="--")
plot_se_traj(ax[1,1], po_latent_nopen[:, 3], "orange", history_nopen.states.est.epistemic_std[:,3], "no penalty")
ax[1,1].spines["top"].set_visible(False)
ax[1,1].spines["right"].set_visible(False)
ax[1,1].grid(visible=True, alpha=0.2)
ax[1,1].set_title("State Estimates with no penalty stepper planner")
#Constant Excitation plot
render(angles_excit, ax[2, 0])
ax[2,1].plot(jnp.arange(n_steps), jnp.full((n_steps,), fill_value=po_states_excit.true.mass[0]), linestyle="--")
plot_se_traj(ax[2,1], po_latent_excit[:,3], "orange", history_excit.states.est.epistemic_std[:,3], "excitation")
ax[2,1].spines["top"].set_visible(False)
ax[2,1].spines["right"].set_visible(False)
ax[2,1].grid(visible=True, alpha=0.2)
ax[2,1].set_title("State Estimates with no penalty stepper planner with constant excitation")
fig.suptitle(f"Comparison with {"random" if not start_upright else "upright"} start")
plt.tight_layout()
plt.show()

#Plot SEs in one plot
fig, ax = plt.subplots(figsize=(10,10))
ax.plot(jnp.arange(n_steps), jnp.full((n_steps,), fill_value=po_states_nopen.true.mass[0]), linestyle="--")
plot_se_traj(ax, po_latent_epipen[:, 3], "blue", history_epipen.states.est.epistemic_std[:,3], "epistemic penalty")
plot_se_traj(ax, po_latent_nopen[:,3], "orange", history_nopen.states.est.epistemic_std[:,3], "no penalty")
plot_se_traj(ax, po_latent_excit[:,3], "green", history_excit.states.est.epistemic_std[:,3], "const. excitation")
ax.plot(jnp.arange(n_steps), po_latent_excit[:, 3], color="green", label="const. excitation")
ax.set_title("State Estimation differences")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(visible=True, alpha=0.2)
ax.legend()
plt.show()

#Plot Controls difference!
#Need to clip them to show actual controls!
controls_epipen = jnp.clip(history_epipen.controls, -mdp.max_torque, mdp.max_torque)
controls_nopen = jnp.clip(history_nopen.controls, -mdp.max_torque, mdp.max_torque)
controls_excit = jnp.clip(history_excit.controls, -mdp.max_torque, mdp.max_torque)
fig, ax = plt.subplots(figsize=(10,10))
ax.plot(jnp.arange(n_steps), controls_epipen, label="epistemic penalty")
ax.plot(jnp.arange(n_steps), controls_nopen, label="no penalty")
ax.plot(jnp.arange(n_steps), controls_excit, label="const. excitation")
ax.set_title("Controls in each Policy")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(visible=True, alpha=0.2)
ax.legend()
plt.show()

#Plot costs
fig, ax = plt.subplots(figsize=(10,10))
ax.plot(jnp.arange(n_steps), history_epipen.costs, label="epistemic_penalty")
ax.plot(jnp.arange(n_steps), history_nopen.costs, label="no penalty")
ax.plot(jnp.arange(n_steps), history_excit.costs, label="const. excitation")
ax.set_title("Costs in each Policy")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(visible=True, alpha=0.2)
ax.legend()
plt.show()

#Multimass plot
results_epipen = dict()
results_nopen = dict()
results_excit = dict()

for mass in masses:
    ini_state = wrapped_mdp.init(jr.PRNGKey(int(mass*100)))
    ini_state = change_state_mass(ini_state, mass)
    upright_state = make_state_upright(ini_state)

    state = upright_state if start_upright else ini_state

    history_epipen = simulate(
        mdp=wrapped_mdp,
        policy=policy_epipen,
        n_steps=n_steps,
        key=jr.PRNGKey(200),
        initial_state=state,
    )
    history_nopen = simulate(
        mdp=wrapped_mdp,
        policy=policy_nopen,
        n_steps=n_steps,
        key=jr.PRNGKey(200),
        initial_state=state,
    )
    history_excit = simulate(
        mdp=wrapped_mdp,
        policy=policy_const_excitation,
        n_steps=n_steps,
        key=jr.PRNGKey(200),
        initial_state=state,
    )

    po_states_epipen = history_epipen.states.obs
    po_latent_epipen = history_epipen.states.latent

    po_states_nopen = history_nopen.states.obs
    po_latent_nopen = history_nopen.states.latent

    po_states_excit = history_excit.states.obs
    po_latent_excit = history_excit.states.latent

    angles_epipen = po_states_epipen.true.angle_normed
    angles_nopen = po_states_nopen.true.angle_normed
    angles_excit = po_states_excit.true.angle_normed

    results_epipen[mass] = {"loc": po_latent_epipen[:,3], "epi": history_epipen.states.est.epistemic_std[:,3], "angles": angles_epipen}
    results_nopen[mass] = {"loc": po_latent_nopen[:,3], "epi": history_nopen.states.est.epistemic_std[:,3], "angles": angles_nopen}
    results_excit[mass] = {"loc": po_latent_excit[:,3], "epi": history_excit.states.est.epistemic_std[:,3], "angles": angles_excit}


#PLOTS
#Plot estimates per mass:
fig, ax = plt.subplots(2, 4, figsize=(20,10))
for i, mass in enumerate(masses):
    row, col = divmod(i, 4)

    ax[row, col].plot(range(n_steps), jnp.full((n_steps,), fill_value=mass), linestyle="--", color="black")
    plot_se_traj(ax[row, col], results_epipen[mass]["loc"], "blue", results_epipen[mass]["epi"], "epistemic penalty")
    plot_se_traj(ax[row, col], results_nopen[mass]["loc"], "orange", results_nopen[mass]["epi"], "no penalty")
    plot_se_traj(ax[row, col], results_excit[mass]["loc"], "green", results_excit[mass]["epi"], "const. excitation")
    ax[row, col].grid(visible=True, alpha=0.2)
    ax[row, col].spines["top"].set_visible(False)
    ax[row, col].spines["right"].set_visible(False)
ax[0, 0].legend()
fig.suptitle("Estimates per mass")
plt.show()


#Plot MAE over time
loc_shape = results_epipen[mass]["loc"].shape[0]
fig, ax = plt.subplots(2, 4, figsize=(20,10))
for i, mass in enumerate(masses):
    row, col = divmod(i, 4)

    epipen_raw = jr.normal(jr.PRNGKey(420), shape=(100, loc_shape))
    epipen_samples = results_epipen[mass]["loc"][None, :] + results_epipen[mass]["epi"][None, :] * epipen_raw
    nopen_raw = jr.normal(jr.PRNGKey(240), shape=(100, loc_shape))
    nopen_samples = results_nopen[mass]["loc"][None, :] + results_nopen[mass]["epi"][None, :] * nopen_raw
    excit_raw = jr.normal(jr.PRNGKey(204), shape=(100, loc_shape))
    excit_samples = results_excit[mass]["loc"][None, :] + results_excit[mass]["epi"][None, :] * excit_raw

    ax[row, col].plot(range(n_steps), jnp.mean(jnp.abs(epipen_samples-mass), axis=0), color="blue", label="epistemic penalty")
    ax[row, col].plot(range(n_steps), jnp.mean(jnp.abs(nopen_samples-mass), axis=0), color="orange", label="no penalty")
    ax[row, col].plot(range(n_steps), jnp.mean(jnp.abs(excit_samples-mass), axis=0), color="green", label="excitation")
    ax[row, col].set_title(f"Mass: {mass}")
    ax[row, col].spines["top"].set_visible(False)
    ax[row, col].spines["right"].set_visible(False)
    ax[row, col].grid(visible=True, alpha=0.2)
ax[0,0].legend()
fig.suptitle("MAE 100 samples per step (loc=se.mean, std=se.epistemic_uncertainty)")
plt.show()


#Plot STD over time
fig, ax = plt.subplots(2, 4, figsize=(20,10))
for i, mass in enumerate(masses):
    row, col = divmod(i, 4)

    ax[row, col].plot(range(n_steps), jnp.abs(results_epipen[mass]["epi"]), color="blue", label="UNC epistemic penalty")
    ax[row, col].plot(range(n_steps), jnp.abs(results_nopen[mass]["epi"]), color="orange", label="UNC no penalty")
    ax[row, col].plot(range(n_steps), jnp.abs(results_excit[mass]["epi"]), color="green", label="UNC excitation")
    ax[row, col].set_title(f"Mass: {mass}")
    ax[row, col].spines["top"].set_visible(False)
    ax[row, col].spines["right"].set_visible(False)
    ax[row, col].grid(visible=True, alpha=0.2)
ax[0,0].legend()
fig.suptitle("Epistemic Uncertainty over time")
plt.show()
