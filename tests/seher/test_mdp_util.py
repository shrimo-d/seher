import jax
import jax.numpy as jnp
import jax.random as jr

from flax.struct import dataclass
from seher.systems.pendulum import Pendulum, PendulumState
from seher.mdp_util import NoiseWrapperState, NoiseWrapperMDP, WorldModelMDP
from seher.models.world_model import WorldModel, WorldModelEnsemble

#Helpers
@dataclass
class RandomPolicy:
    mdp: object

    def __call__(self, carry, obs, control, key):
        return None, jr.uniform(
            key,
            shape=self.mdp.empty_control().shape,
            minval=self.mdp.control_min,
            maxval=self.mdp.control_max,
        )

    def initial_carry(self):
        return None

def pendulum_array_to_state(arr: jax.Array):
    angle = jnp.arctan2(arr[..., 1], arr[..., 0])
    velocity = arr[..., -1]
    return PendulumState(angle=angle, velocity=velocity)

def pendulum_array_to_state_noisy(arr: jax.Array):
    s = pendulum_array_to_state(arr)
    return NoiseWrapperState(original_state=s, noisy_state=s)

def pendulum_add_noise(state: PendulumState, key):
    key_ang, key_vel = jr.split(key)
    ang_noise = jr.normal(key_ang, ())
    vel_noise = jr.normal(key_vel, ())
    return PendulumState(
        angle=state.angle + 0.5 * ang_noise,
        velocity=state.velocity + 0.5 * vel_noise,
    )

def universal_state_to_array(state):
    state = getattr(state, "noisy_state", state)
    return jnp.array([jnp.cos(state.angle), jnp.sin(state.angle), state.velocity])

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

def test_world_model_mdp_transit_and_cost_monotonicity():
    mdp = Pendulum()
    noisy = NoiseWrapperMDP(mdp=mdp, add_noise_to_state=pendulum_add_noise)

    ens = WorldModelEnsemble.create(
        n_models=5,
        key=jr.PRNGKey(0),
        inpt_size=4,
        layer_sizes=[16],
        output_size=3,
        activations=[jax.nn.tanh, lambda x: x],
        state_to_array=universal_state_to_array,
        control_to_array=lambda x: x,
        state_dim=3,
        control_dim=1,
    )

    wm0 = WorldModelMDP(
        original_mdp=noisy,
        model=ens,
        array_to_state=pendulum_array_to_state_noisy,
        uncertainty_weight=0.0,
    )
    wm1 = WorldModelMDP(
        original_mdp=noisy,
        model=ens,
        array_to_state=pendulum_array_to_state_noisy,
        uncertainty_weight=1.0,
    )

    s = noisy.init(jr.PRNGKey(1))
    u = mdp.empty_control()

    s_next = wm0.transit(s, u, jr.PRNGKey(2))
    assert isinstance(s_next, NoiseWrapperState)

    c0 = wm0.cost(s,u, jr.PRNGKey(3))
    c1 = wm1.cost(s,u, jr.PRNGKey(3))
    assert c1 >= c0 - 1e-6