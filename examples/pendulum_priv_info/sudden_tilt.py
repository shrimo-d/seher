import jax
import jax.numpy as jnp
import jax.random as jr

import functools
import optax

import matplotlib.pyplot as plt

from flax.struct import dataclass
from pathlib import Path

from seher.models.state_estimator import (
    FeatureEnsembleLatent,
    StateEstimatorMDP,
)

from seher.systems.pendulum import render
from seher.systems.pendulum_po import (
    PartiallyObservablePendulum,
    POPendulumTrueState,
    POPendulumObservation,
    POPendulumState,
)

from seher.control.stepper_planner import StepperPlanner
from seher.stepper.optax import OptaxOptimizer
from seher.ars import ars_value_and_grad
from seher.control.mpc import MPCPolicy

from planning_mdp import EstimatedPendulumPlanningMDP
from load_models import maybe_load_estimator


# ============================================================
# CONFIG
# ============================================================

mass = 1.2
n_steps = 250

epistemic_penalty = 15.0

mass_idx = 3

# IMPORTANT:
# all policies get tilted at EXACTLY this timestep
tilt_step = 75

tilt_std = 1.2


# ============================================================
# CONSTANT EXCITATION POLICY
# ============================================================

@dataclass
class ConstantExcitationMPCPolicy:
    mpc: MPCPolicy
    noise_std: float = 0.7

    def initial_carry(self):
        return self.mpc.initial_carry()

    def __call__(self, carry, obs, control, key):

        k_pol, k_noise = jr.split(key)

        carry, action = self.mpc(
            carry,
            obs,
            control,
            k_pol,
        )

        noise = (
            jr.normal(
                k_noise,
                shape=action.shape,
            )
            * self.noise_std
        )

        return carry, action + noise


# ============================================================
# ENVIRONMENT
# ============================================================

mdp = PartiallyObservablePendulum(
    min_mass=0.5,
    max_mass=2.5,
)

gru = maybe_load_estimator(
    Path("./examples/pendulum_priv_info"),
    "det_gru_ensemble",
)

wrapped_mdp = StateEstimatorMDP(
    original_mdp=mdp,
    estimator=gru,
    adapter=FeatureEnsembleLatent(latent_dim=12),
)


# ============================================================
# PLANNING MDPS
# ============================================================

plan_mdp_epipen = EstimatedPendulumPlanningMDP(
    mdp=mdp,
    epistemic_penalty=epistemic_penalty,
    estimator=gru,
    adapter=FeatureEnsembleLatent(latent_dim=12),
)

plan_mdp_nopen = EstimatedPendulumPlanningMDP(
    mdp=mdp,
    epistemic_penalty=0.0,
    estimator=gru,
    adapter=FeatureEnsembleLatent(latent_dim=12),
)


# ============================================================
# PLANNERS
# ============================================================

def make_planner(plan_mdp):

    return StepperPlanner(
        mdp=plan_mdp,
        n_iter=13,
        n_plan_steps=20,
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


planner_epipen = make_planner(plan_mdp_epipen)
planner_nopen = make_planner(plan_mdp_nopen)


# ============================================================
# POLICIES
# ============================================================

policy_epipen = MPCPolicy(
    mdp=plan_mdp_epipen,
    planner=planner_epipen,
)

policy_nopen = MPCPolicy(
    mdp=plan_mdp_nopen,
    planner=planner_nopen,
)

policy_excit = ConstantExcitationMPCPolicy(
    mpc=policy_nopen,
    noise_std=0.7,
)


# ============================================================
# INITIAL STATE
# ============================================================

ini_state = wrapped_mdp.init(jr.PRNGKey(0))

initial_state = ini_state.replace(
    obs=POPendulumState(
        true=POPendulumTrueState(
            angle=jnp.array((0.0,)),
            velocity=jnp.array((0.0,)),
            mass=jnp.array((mass,)),
        ),
        obs=POPendulumObservation(
            angle=jnp.array((0.0,)),
            velocity=jnp.array((0.0,)),
        ),
    )
)


# ============================================================
# RANDOM TILT
# ============================================================

def apply_random_tilt(
    wrapped_mdp,
    state,
    prev_control,
    key,
    tilt_std=1.0,
):

    k_angle, k_est, k_latent = jr.split(key, 3)

    delta_angle = (
        jr.normal(
            k_angle,
            shape=state.obs.true.angle.shape,
        )
        * tilt_std
    )

    # --------------------------------------------------------
    # perturb BOTH true state and observation
    # --------------------------------------------------------

    tilted_obs = state.obs.replace(
        true=state.obs.true.replace(
            angle=state.obs.true.angle + delta_angle,
        ),
        obs=state.obs.obs.replace(
            angle=state.obs.obs.angle + delta_angle,
        ),
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # preserve estimator carry/history
    # --------------------------------------------------------

    se_carry_new, est_new = wrapped_mdp.estimator(
        state.se_carry,
        tilted_obs,
        prev_control,
        k_est,
    )

    latent_new = wrapped_mdp.adapter(
        est_new,
        k_latent,
    )

    return state.replace(
        obs=tilted_obs,
        se_carry=se_carry_new,
        est=est_new,
        latent=latent_new,
    )


# ============================================================
# CUSTOM ROLLOUT
# ============================================================

def rollout_with_sudden_tilt(
    wrapped_mdp,
    policy,
    initial_state,
    key,
    n_steps,
    tilt_step,
):

    def step(carry, t):

        (
            state,
            pol_carry,
            prev_control,
            already_tilted,
            key,
        ) = carry

        key, k_pol, k_env, k_tilt = jr.split(key, 4)

        # ----------------------------------------------------
        # fixed tilt step for ALL policies
        # ----------------------------------------------------

        should_tilt = (
            (~already_tilted)
            & (t == tilt_step)
        )

        tilted_state = apply_random_tilt(
            wrapped_mdp=wrapped_mdp,
            state=state,
            prev_control=prev_control,
            key=k_tilt,
            tilt_std=tilt_std,
        )

        state = jax.lax.cond(
            should_tilt,
            lambda _: tilted_state,
            lambda _: state,
            operand=None,
        )

        already_tilted = (
            already_tilted
            | should_tilt
        )

        # ----------------------------------------------------
        # policy step
        # ----------------------------------------------------

        pol_carry, control = policy(
            pol_carry,
            state,
            prev_control,
            k_pol,
        )

        next_state = wrapped_mdp.transit(
            state,
            control,
            k_env,
        )

        cost = wrapped_mdp.cost(
            state,
            control,
            k_env,
        )

        next_carry = (
            next_state,
            pol_carry,
            control,
            already_tilted,
            key,
        )

        outs = (
            next_state,
            control,
            cost,
            should_tilt,
            state.est.loc[mass_idx],
        )

        return next_carry, outs

    init_carry = (
        initial_state,
        policy.initial_carry(),
        wrapped_mdp.empty_control(),
        jnp.array(False),
        key,
    )

    _, outputs = jax.lax.scan(
        step,
        init_carry,
        jnp.arange(n_steps),
    )

    return outputs


# ============================================================
# RUN ALL POLICIES
# ============================================================

results = {}

for name, pol in [
    ("epistemic penalty", policy_epipen),
    ("no penalty", policy_nopen),
    ("constant excitation", policy_excit),
]:

    results[name] = rollout_with_sudden_tilt(
        wrapped_mdp=wrapped_mdp,
        policy=pol,
        initial_state=initial_state,
        key=jr.PRNGKey(43),
        n_steps=n_steps,
        tilt_step=tilt_step,
    )


# ============================================================
# COLORS
# ============================================================

colors = {
    "epistemic penalty": "blue",
    "no penalty": "orange",
    "constant excitation": "green",
}


# ============================================================
# MASS ESTIMATES
# ============================================================

fig, ax = plt.subplots(figsize=(12, 5))

ax.axhline(
    mass,
    linestyle="--",
    color="black",
    label="true mass",
)

for name, out in results.items():

    states, controls, costs, tilt_events, mass_estimates = out

    epi_std = states.est.epistemic_std[:, mass_idx]

    ax.plot(
        mass_estimates,
        color=colors[name],
        label=name,
    )

    ax.fill_between(
        jnp.arange(n_steps),
        mass_estimates - epi_std,
        mass_estimates + epi_std,
        color=colors[name],
        alpha=0.2,
    )

ax.axvline(
    tilt_step,
    color="red",
    linestyle="--",
    label="sudden tilt",
    alpha=0.3
)
ax.grid(visible=True, alpha=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("Mass estimates after sudden tilt")
ax.legend()

plt.show()


# ============================================================
# ANGLES
# ============================================================

fig, ax = plt.subplots(figsize=(12, 5))

for name, out in results.items():

    states, controls, costs, tilt_events, mass_estimates = out

    angles = states.obs.true.angle_normed[:, 0]

    ax.plot(
        angles,
        color=colors[name],
        label=name,
    )

ax.axvline(
    tilt_step,
    color="red",
    linestyle="--",
    label="sudden tilt",
    alpha=0.3
)
ax.grid(visible=True, alpha=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("Pendulum angle")
ax.legend()

plt.show()


# ============================================================
# CONTROLS
# ============================================================

fig, ax = plt.subplots(figsize=(12, 5))

for name, out in results.items():

    states, controls, costs, tilt_events, mass_estimates = out

    ax.plot(
        jnp.clip(
            controls,
            -mdp.max_torque,
            mdp.max_torque,
        ),
        color=colors[name],
        label=name,
    )

ax.axvline(
    tilt_step,
    color="red",
    linestyle="--",
    alpha=0.3
)

ax.set_title("Controls")
ax.grid(visible=True, alpha=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend()

plt.show()


# ============================================================
# COSTS
# ============================================================

fig, ax = plt.subplots(figsize=(12, 5))

for name, out in results.items():

    states, controls, costs, tilt_events, mass_estimates = out

    ax.plot(
        costs,
        color=colors[name],
        label=name,
    )

ax.axvline(
    tilt_step,
    color="red",
    linestyle="--",
    alpha=0.3
)

ax.set_title("Costs")
ax.grid(visible=True, alpha=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend()

plt.show()

fig = plt.figure(figsize=(12, 8))

outer = fig.add_gridspec(
    3,
    1,
    hspace=0.25,
)

policy_order = [
    ("epistemic penalty", "blue"),
    ("no penalty", "orange"),
    ("constant excitation", "green"),
]

for row, (name, color) in enumerate(policy_order):

    states, controls, costs, tilt_events, mass_estimates = results[name]

    angles = states.obs.true.angle_normed[:, 0]

    ax = fig.add_subplot(outer[row, 0])

    # --------------------------------------------------------
    # render pendulum trajectory
    # --------------------------------------------------------

    render(angles, ax)

    # --------------------------------------------------------
    # sudden tilt marker
    # --------------------------------------------------------

    ax.axvline(
        tilt_step,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.3
    )

    ax.set_title(name)

plt.suptitle("Pendulum trajectories with sudden tilt")

plt.show()