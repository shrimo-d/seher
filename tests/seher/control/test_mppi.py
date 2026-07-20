"""Tests for MPPI planning."""

import jax.numpy as jnp
import jax.random as jr

from seher.control.stepper_planner import StepperPlanner
from seher.stepper.mppi import GaussianMPPIOptimizer
from seher.systems.pendulum import Pendulum


def test_one_step():
    """Test a single step of MPPI planning numerically."""
    key = jr.PRNGKey(32)
    mdp = Pendulum()
    planner = StepperPlanner(
        mdp=mdp,
        n_iter=2,
        n_plan_steps=2,
        optimizer=GaussianMPPIOptimizer(
            initial_loc=jnp.array(0.0),
            initial_scale=jnp.array(1.0),
            objective=None,
            n_candidates=3,
            top_k=2,
        ),
    )
    carry = planner(
        state=mdp.init(key=key), carry=planner.initial_carry(), key=key
    )
    assert jnp.allclose(carry.plan[0], jnp.array([[-0.8427123]]))


def test_default_temperature_produces_finite_plan():
    """A zero temperature gives uniform, finite candidate weights."""
    key = jr.PRNGKey(12)
    mdp = Pendulum()
    planner = StepperPlanner(
        mdp=mdp,
        n_iter=1,
        n_plan_steps=2,
        optimizer=GaussianMPPIOptimizer(
            initial_loc=jnp.array(0.0),
            initial_scale=jnp.array(1.0),
            objective=None,
            n_candidates=4,
            top_k=2,
        ),
    )

    carry = planner(
        state=mdp.init(key=key), carry=planner.initial_carry(), key=key
    )

    assert jnp.all(jnp.isfinite(carry.plan))
