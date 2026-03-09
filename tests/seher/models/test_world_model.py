import jax
import jax.numpy as jnp
import jax.random as jr
from seher.apx_arch import MLP
from seher.models.world_model import WorldModel, WorldModelEnsemble, WorldModelMDP
from seher.systems.pendulum import Pendulum, PendulumState
from seher.mdp_util import NoiseWrapperMDP, NoiseWrapperState
from seher.jax_util import tree_stack

#Helpers
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

def test_world_model_forward_shapes():
    key = jr.PRNGKey(0)
    mlp = MLP.make(
        inpt_size=4,
        layer_sizes=[8],
        output_size=3,
        activations=[jax.nn.tanh, lambda x: x],
        key=key,
    )
    wm = WorldModel(mlp=mlp, 
                    state_to_array=universal_state_to_array,
                    control_to_array=lambda x: x)

    s = jnp.zeros((10,3))
    u = jnp.zeros((10,1))
    y = wm(s, u, key)

    assert y.shape == (10,3)

def test_ensemble_shapes_and_determinism_for_same_key():
    key = jr.PRNGKey(0)
    ens = WorldModelEnsemble.create(
        n_models=5,
        key=key,
        inpt_size=4,
        layer_sizes=[16,16],
        output_size=3,
        activations=[jax.nn.tanh, jax.nn.tanh, lambda x: x],
        state_to_array=universal_state_to_array,
        control_to_array=lambda x: x,
        state_dim=3,
        control_dim=1,
    )
    mdp = Pendulum()
    s = mdp.init(jr.PRNGKey(1))
    u = mdp.empty_control()

    pred1, std1 = ens(s, u, jr.PRNGKey(123))
    pred2, std2 = ens(s, u, jr.PRNGKey(123))

    assert pred1.shape[-1] == 3
    assert std1.shape[-1] == 3

    assert jnp.allclose(pred1, pred2)
    assert jnp.allclose(std1, std2)

def test_ensemble_std_zero_if_models_identical():
    key = jr.PRNGKey(0)
    mlp = MLP.make(
        inpt_size=4,
        layer_sizes=[8],
        output_size=3,
        activations=[jax.nn.tanh, lambda x: x],
        key=key,
    )
    m1 = WorldModel(
        mlp=mlp,
        state_to_array=universal_state_to_array,
        control_to_array=lambda x: x,
    )
    m2 = WorldModel(
        mlp=mlp,
        state_to_array=universal_state_to_array,
        control_to_array=lambda x: x,
    )
    m3 = WorldModel(
        mlp=mlp,
        state_to_array=universal_state_to_array,
        control_to_array=lambda x: x,
    )
    models = tree_stack([m1, m2, m3])

    ens = WorldModelEnsemble(
        models=models,
        n_models=3,
        state_to_array=universal_state_to_array,
        control_to_array=lambda x: x,
        state_dim=3,
        control_dim=1,
    )

    mdp = Pendulum()
    s = mdp.init(jr.PRNGKey(1))
    u = mdp.empty_control()
    pred, std = ens(s, u, jr.PRNGKey(2))

    assert jnp.allclose(std, 0.0, atol=1e-6)

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