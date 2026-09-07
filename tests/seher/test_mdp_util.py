import jax
import jax.numpy as jnp
import jax.random as jr

from seher.systems.pendulum import Pendulum, PendulumState
from seher.mdp_util import NoiseWrapperMDP
from seher.models.random_policy import RandomPolicy

#Helpers
def pendulum_add_noise(state: PendulumState, key):
    key_ang, key_vel = jr.split(key)
    ang_noise = jr.normal(key_ang, ())
    vel_noise = jr.normal(key_vel, ())
    return PendulumState(
        angle=state.angle + 0.5 * ang_noise,
        velocity=state.velocity + 0.5 * vel_noise,
    )

#Tests
def test_noise_wrapper_state_structure_and_cost_passthrough():
    mdp = Pendulum()
    noisy = NoiseWrapperMDP(mdp=mdp, add_noise_to_state=pendulum_add_noise)
    rp = RandomPolicy(mdp=mdp)

    key = jr.PRNGKey(0)
    s0 = noisy.init(key)
    assert hasattr(s0, "original_state")
    assert hasattr(s0, "noisy_state")

    u = mdp.empty_control()
    c_wrapped = noisy.cost(s0, u, jr.PRNGKey(1))
    c_base = mdp.cost(s0.original_state, u, jr.PRNGKey(1))
    assert jnp.allclose(c_wrapped, c_base)