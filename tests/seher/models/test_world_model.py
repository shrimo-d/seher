import jax
import jax.numpy as jnp
import jax.random as jr
from seher.apx_arch import MLP
from seher.models.world_model import WorldModel, WorldModelEnsemble
from seher.systems.pendulum import Pendulum
from seher.jax_util import tree_stack

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

