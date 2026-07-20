import functools

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.struct import dataclass, field

from seher.ars import ars_value_and_grad
from seher.control.mpc import MPCPolicy
from seher.control.stepper_planner import StepperPlanner
from seher.stepper.mppi import GaussianMPPIOptimizer
from seher.stepper.optax import OptaxOptimizer

from controller_presets import ARS_REFERENCE, MPPI_STANDARD, ControllerConfig
from pendulum_planner import PendulumPlanner


@dataclass
class ConstantExcitationPolicy:
    policy: object
    mdp: object = field(pytree_node=False)
    excitation: object

    def initial_carry(self):
        return self.policy.initial_carry()

    def __call__(self, carry, obs, control, key):
        carry, planned_control = self.policy(carry, obs, control, key)
        excited_control = planned_control + jnp.asarray(
            self.excitation,
            dtype=planned_control.dtype,
        )
        return carry, jnp.clip(
            excited_control,
            self.mdp.control_min,
            self.mdp.control_max,
        )


def create_ars_optimizer(
    lr=ARS_REFERENCE.learning_rate,
    std=ARS_REFERENCE.ars_std,
    n_perturbations=ARS_REFERENCE.n_perturbations,
    top_k=ARS_REFERENCE.top_k,
):
    return OptaxOptimizer(
        objective=None,
        optimizer=optax.adam(lr),
        value_and_grad=functools.partial(
            ars_value_and_grad,
            std=std,
            n_perturbations=n_perturbations,
            top_k=top_k,
        ),
    )


def create_mppi_optimizer(
    n_candidates=MPPI_STANDARD.mppi_candidates,
    top_k=MPPI_STANDARD.mppi_top_k,
    initial_loc=0.0,
    initial_scale=MPPI_STANDARD.mppi_initial_scale,
    min_scale=MPPI_STANDARD.mppi_min_scale,
    temperature=MPPI_STANDARD.mppi_temperature,
):
    return GaussianMPPIOptimizer(
        objective=None,
        n_candidates=n_candidates,
        top_k=top_k,
        initial_loc=jnp.array(initial_loc),
        initial_scale=jnp.array(initial_scale),
        min_scale=min_scale,
        temperature=temperature,
    )


def create_optimizer(
    optimizer,
    *,
    learning_rate=ARS_REFERENCE.learning_rate,
    ars_std=ARS_REFERENCE.ars_std,
    n_perturbations=ARS_REFERENCE.n_perturbations,
    top_k=ARS_REFERENCE.top_k,
    n_candidates=MPPI_STANDARD.mppi_candidates,
    mppi_top_k=MPPI_STANDARD.mppi_top_k,
    initial_loc=0.0,
    initial_scale=MPPI_STANDARD.mppi_initial_scale,
    min_scale=MPPI_STANDARD.mppi_min_scale,
    temperature=MPPI_STANDARD.mppi_temperature,
):
    if optimizer == "ars":
        if top_k > n_perturbations:
            raise ValueError("ARS top_k must not exceed n_perturbations.")
        return create_ars_optimizer(
            lr=learning_rate,
            std=ars_std,
            n_perturbations=n_perturbations,
            top_k=top_k,
        )
    if optimizer == "mppi":
        if mppi_top_k > n_candidates:
            raise ValueError("MPPI top_k must not exceed n_candidates.")
        return create_mppi_optimizer(
            n_candidates=n_candidates,
            top_k=mppi_top_k,
            initial_loc=initial_loc,
            initial_scale=initial_scale,
            min_scale=min_scale,
            temperature=temperature,
        )
    raise ValueError(f"Unknown optimizer: {optimizer}")


def create_optimizer_from_config(config: ControllerConfig):
    """Build the optimizer described by a fully resolved controller config."""

    config.validated()
    return create_optimizer(
        config.optimizer,
        learning_rate=config.learning_rate,
        ars_std=config.ars_std,
        n_perturbations=config.n_perturbations,
        top_k=config.top_k,
        n_candidates=config.mppi_candidates,
        mppi_top_k=config.mppi_top_k,
        initial_scale=config.mppi_initial_scale,
        min_scale=config.mppi_min_scale,
        temperature=config.mppi_temperature,
    )


def create_mpc_policy(mdp, n_iter, n_plan_steps, optimizer):
    stepper = StepperPlanner(
        mdp=mdp,
        n_iter=n_iter,
        n_plan_steps=n_plan_steps,
        warm_start=True,
        optimizer=optimizer,
    )

    planner = stepper
    if hasattr(mdp, "prepare_planning_state"):
        planner = PendulumPlanner(
            planner=stepper,
            prepare_state=mdp.prepare_planning_state,
        )
    return MPCPolicy(mdp=mdp, planner=planner)


def create_mpc_policy_from_config(mdp, config: ControllerConfig):
    """Build an MPC policy from one fully resolved pendulum controller config."""

    return create_mpc_policy(
        mdp,
        n_iter=config.n_iter,
        n_plan_steps=config.n_plan_steps,
        optimizer=create_optimizer_from_config(config),
    )


def make_constant_excitation(mdp, value=0.0, vector=None):
    empty_control = mdp.empty_control()
    if vector is not None:
        excitation = jnp.asarray(vector)
        if excitation.shape != empty_control.shape:
            raise ValueError(
                f"excitation vector must have shape {empty_control.shape}, "
                f"got {excitation.shape}"
            )
        return excitation
    return jnp.full_like(empty_control, value)


def maybe_add_constant_excitation(policy, mdp, excitation):
    if bool(np.asarray(jax.device_get(jnp.all(excitation == 0.0)))):
        return policy
    return ConstantExcitationPolicy(
        policy=policy,
        mdp=mdp,
        excitation=excitation,
    )
