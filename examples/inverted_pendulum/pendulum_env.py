"""Small state helpers for the partially observable inverted pendulum."""

from __future__ import annotations

import jax.numpy as jnp

from seher.systems.pendulum_po import POPendulumState


def replace_physical_state(
    state: POPendulumState,
    *,
    angle=None,
    velocity=None,
    mass=None,
) -> POPendulumState:
    """Return a consistent state after replacing selected physical values."""

    true = state.true
    if angle is not None:
        true = true.replace(angle=jnp.asarray(angle).reshape((1,)))
    if velocity is not None:
        true = true.replace(velocity=jnp.asarray(velocity).reshape((1,)))
    if mass is not None:
        true = true.replace(mass=jnp.asarray(mass).reshape((1,)))

    observation = state.obs.replace(
        angle=true.angle,
        velocity=true.velocity,
    )
    return state.replace(true=true, obs=observation)


def make_upright(state: POPendulumState) -> POPendulumState:
    """Place a pendulum state at the zero-angle, zero-velocity target."""

    return replace_physical_state(state, angle=0.0, velocity=0.0)


def replace_wrapped_physical_state(state, **changes):
    """Apply :func:`replace_physical_state` to a StateEstimatorMDP state."""

    return state.replace(obs=replace_physical_state(state.obs, **changes))


def make_wrapped_upright(state):
    """Place a wrapped pendulum state at the upright target."""

    return state.replace(obs=make_upright(state.obs))
