import jax.numpy as jnp
import jax.random as jr

from seher.systems.pendulum_po import (
    POPendulumTrueState,
    POPendulumObservation,
    PartiallyObservablePendulum,
    POPendulumState,
)

def test_true_state_cos_sin_repr_contains_cos_sin_velocity_and_mass():
    state = POPendulumTrueState(
        angle=jnp.array([jnp.pi / 2]),
        velocity=jnp.array([1.5]),
        mass=jnp.array([2.0]),
    )
    actual = state.cos_sin_repr()
    expected = jnp.array([0.0, 1.0, 1.5, 2.0])
    assert jnp.allclose(actual, expected, atol=1e-6)


def test_observation_cos_sin_repr_contains_cos_sin_and_velocity_only():
    obs = POPendulumObservation(angle=jnp.array([jnp.pi]), velocity=jnp.array([-2.0]))
    actual = obs.cos_sin_repr()
    expected = jnp.array([-1.0, 0.0, -2.0])
    assert jnp.allclose(actual, expected, atol=1e-6)


def test_angle_normed_maps_angles_into_closed_open_interval_minus_pi_to_pi():
    true_state = POPendulumTrueState(angle=jnp.array([3 * jnp.pi]), velocity=jnp.array([0.0]), mass=jnp.array([1.0]))
    obs = POPendulumObservation(angle=jnp.array([-3 * jnp.pi]), velocity=jnp.array([0.0]))
    assert jnp.allclose(true_state.angle_normed, jnp.array([-jnp.pi]))
    assert jnp.allclose(obs.angle_normed, jnp.array([-jnp.pi]))


def test_control_bounds_match_max_torque():
    env = PartiallyObservablePendulum(max_torque=2.5)
    assert jnp.array_equal(env.control_min, jnp.array([-2.5]))
    assert jnp.array_equal(env.control_max, jnp.array([2.5]))


def test_init_returns_consistent_observation_and_true_state_within_ranges():
    env = PartiallyObservablePendulum()
    state = env.init(jr.PRNGKey(0))

    assert isinstance(state, POPendulumState)
    assert state.true.angle.shape == (1,)
    assert state.true.velocity.shape == (1,)
    assert state.true.mass.shape == (1,)
    assert jnp.array_equal(state.true.angle, state.obs.angle)
    assert jnp.array_equal(state.true.velocity, state.obs.velocity)
    assert -jnp.pi <= float(state.true.angle[0]) <= jnp.pi
    assert -8.0 <= float(state.true.velocity[0]) <= 8.0
    assert 0.5 <= float(state.true.mass[0]) <= 4.0


def test_init_is_deterministic_for_same_key():
    env = PartiallyObservablePendulum()
    state1 = env.init(jr.PRNGKey(1))
    state2 = env.init(jr.PRNGKey(1))
    assert jnp.array_equal(state1.true.angle, state2.true.angle)
    assert jnp.array_equal(state1.true.velocity, state2.true.velocity)
    assert jnp.array_equal(state1.true.mass, state2.true.mass)


def test_transit_matches_manual_dynamics_without_clipping():
    env = PartiallyObservablePendulum(gravity=10.0, length=1.0, time_diff=0.05, max_torque=2.0, max_speed=8.0)
    state = POPendulumState(
        true=POPendulumTrueState(angle=jnp.array([0.2]), velocity=jnp.array([0.3]), mass=jnp.array([2.0])),
        obs=POPendulumObservation(angle=jnp.array([0.2]), velocity=jnp.array([0.3])),
    )
    control = jnp.array([1.0])

    next_state = env.transit(state, control, jr.PRNGKey(0))

    angle_acc = -3.0 * env.gravity / (2.0 * env.length) * jnp.sin(state.true.angle + jnp.pi) + 3.0 / (state.true.mass * env.length ** 2.0) * control
    velocity_p1 = state.true.velocity + env.time_diff * angle_acc
    velocity_p1 = jnp.clip(velocity_p1, -env.max_speed, env.max_speed)
    angle_p1 = state.true.angle + env.time_diff * velocity_p1

    assert jnp.allclose(next_state.true.velocity, velocity_p1)
    assert jnp.allclose(next_state.true.angle, angle_p1)
    assert jnp.array_equal(next_state.true.mass, state.true.mass)
    assert jnp.array_equal(next_state.obs.angle, next_state.true.angle)
    assert jnp.array_equal(next_state.obs.velocity, next_state.true.velocity)


def test_transit_clips_control_before_dynamics():
    env = PartiallyObservablePendulum(max_torque=2.0)
    state = POPendulumState(
        true=POPendulumTrueState(angle=jnp.array([0.0]), velocity=jnp.array([0.0]), mass=jnp.array([1.0])),
        obs=POPendulumObservation(angle=jnp.array([0.0]), velocity=jnp.array([0.0])),
    )

    next_with_large = env.transit(state, jnp.array([100.0]), jr.PRNGKey(0))
    next_with_clipped = env.transit(state, jnp.array([2.0]), jr.PRNGKey(1))

    assert jnp.allclose(next_with_large.true.angle, next_with_clipped.true.angle)
    assert jnp.allclose(next_with_large.true.velocity, next_with_clipped.true.velocity)


def test_transit_clips_velocity_to_max_speed():
    env = PartiallyObservablePendulum(time_diff=1.0, max_speed=8.0, max_torque=2.0)
    state = POPendulumState(
        true=POPendulumTrueState(angle=jnp.array([0.0]), velocity=jnp.array([7.9]), mass=jnp.array([0.5])),
        obs=POPendulumObservation(angle=jnp.array([0.0]), velocity=jnp.array([7.9])),
    )

    next_state = env.transit(state, jnp.array([2.0]), jr.PRNGKey(0))
    assert jnp.allclose(next_state.true.velocity, jnp.array([8.0]))


def test_cost_matches_formula_using_normalized_angle():
    env = PartiallyObservablePendulum()
    state = POPendulumState(
        true=POPendulumTrueState(angle=jnp.array([3 * jnp.pi]), velocity=jnp.array([2.0]), mass=jnp.array([1.0])),
        obs=POPendulumObservation(angle=jnp.array([3 * jnp.pi]), velocity=jnp.array([2.0])),
    )
    control = jnp.array([4.0])

    actual = env.cost(state, control, jr.PRNGKey(0))
    expected = jnp.array([jnp.pi ** 2 + 0.1 * 4.0 + 0.001 * 16.0])
    assert jnp.allclose(actual, expected)


def test_empty_control_returns_shape_one_array():
    env = PartiallyObservablePendulum()
    control = env.empty_control()
    assert control.shape == (1,)
