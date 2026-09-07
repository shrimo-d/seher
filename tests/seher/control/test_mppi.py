"""Tests for MPPI planning."""

import jax.numpy as jnp
import jax.random as jr

from seher.control.mpc import MPCPolicy
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
    assert jnp.allclose(carry.plan[0], jnp.array([0.55387306]))


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


def test_mppi_respects_mdp_bounds_through_warm_start_and_execution():
    """Candidates, carry plans, and the executed first action stay bounded."""
    key = jr.PRNGKey(23)
    mdp = Pendulum(max_torque=0.05)
    planner = StepperPlanner(
        mdp=mdp,
        n_iter=2,
        n_plan_steps=3,
        optimizer=GaussianMPPIOptimizer(
            initial_loc=jnp.array(10.0),
            initial_scale=jnp.array(100.0),
            objective=None,
            n_candidates=32,
            top_k=4,
        ),
    )

    initial_carry = planner.initial_carry()
    assert jnp.all(initial_carry.plan >= mdp.control_min)
    assert jnp.all(initial_carry.plan <= mdp.control_max)

    state = mdp.init(key)
    carry = planner(state=state, carry=initial_carry, key=key)
    assert jnp.all(carry.plan >= mdp.control_min)
    assert jnp.all(carry.plan <= mdp.control_max)

    # Even a malformed external carry cannot leak an invalid warm-start plan
    # into candidate evaluation or the returned plan.
    poisoned = carry.replace(
        stepper_carry=carry.stepper_carry.replace(
            current=jnp.full_like(carry.plan, 50.0)
        )
    )
    policy = MPCPolicy(mdp=mdp, planner=planner)
    next_carry, control = policy(
        carry=poisoned,
        obs=state,
        control=mdp.empty_control(),
        key=jr.PRNGKey(24),
    )

    assert jnp.all(next_carry.plan >= mdp.control_min)
    assert jnp.all(next_carry.plan <= mdp.control_max)
    assert jnp.all(control >= mdp.control_min)
    assert jnp.all(control <= mdp.control_max)
    assert jnp.array_equal(control, next_carry.plan[0])
